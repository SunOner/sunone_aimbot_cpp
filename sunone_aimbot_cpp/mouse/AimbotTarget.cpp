#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "sunone_aimbot_cpp.h"
#include "AimbotTarget.h"
#include "config.h"

AimbotTarget::AimbotTarget()
    : x(0), y(0), w(0), h(0), classId(0), pivotX(0.0), pivotY(0.0)
{
}

AimbotTarget::AimbotTarget(int x_, int y_, int w_, int h_, int cls, double px, double py)
    : x(x_), y(y_), w(w_), h(h_), classId(cls), pivotX(px), pivotY(py)
{
}

AimbotTarget* sortTargets(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classes,
    int screenWidth,
    int screenHeight,
    bool disableHeadshot)
{
    if (boxes.empty() || classes.empty())
    {
        return nullptr;
    }

    cv::Point center(screenWidth / 2, screenHeight / 2);

    double minDistance = std::numeric_limits<double>::max();
    int nearestIdx = -1;
    int targetY = 0;

    if (!disableHeadshot)
    {
        for (size_t i = 0; i < boxes.size(); i++)
        {
            if (classes[i] == config.class_head)
            {
                int headOffsetY = static_cast<int>(boxes[i].height * config.head_y_offset);
                cv::Point targetPoint(boxes[i].x + boxes[i].width / 2, boxes[i].y + headOffsetY);
                double distance = std::pow(targetPoint.x - center.x, 2) + std::pow(targetPoint.y - center.y, 2);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearestIdx = static_cast<int>(i);
                    targetY = targetPoint.y;
                }
            }
        }
    }

    if (disableHeadshot || nearestIdx == -1)
    {
        minDistance = std::numeric_limits<double>::max();
        for (size_t i = 0; i < boxes.size(); i++)
        {
            if (disableHeadshot && classes[i] == config.class_head)
                continue;

            if (classes[i] == config.class_player)
            {
                int offsetY = static_cast<int>(boxes[i].height * config.body_y_offset);
                cv::Point targetPoint(boxes[i].x + boxes[i].width / 2, boxes[i].y + offsetY);
                double distance = std::pow(targetPoint.x - center.x, 2) + std::pow(targetPoint.y - center.y, 2);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearestIdx = static_cast<int>(i);
                    targetY = targetPoint.y;
                }
            }
        }
    }

    if (nearestIdx == -1)
    {
        return nullptr;
    }

    int finalY = 0;
    if (classes[nearestIdx] == config.class_head)
    {
        int headOffsetY = static_cast<int>(boxes[nearestIdx].height * config.head_y_offset);
        finalY = boxes[nearestIdx].y + headOffsetY - boxes[nearestIdx].height / 2;
    }
    else
    {
        finalY = targetY - boxes[nearestIdx].height / 2;
    }

    int finalX = boxes[nearestIdx].x;
    int finalW = boxes[nearestIdx].width;
    int finalH = boxes[nearestIdx].height;
    int finalClass = classes[nearestIdx];

    double pivotX = finalX + (finalW / 2.0);
    double pivotY = finalY + (finalH / 2.0);

    return new AimbotTarget(finalX, finalY, finalW, finalH, finalClass, pivotX, pivotY);
}

float MultiTargetTracker::iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float ua = a.width * a.height + b.width * b.height - inter;
    if (ua <= 1e-6f) return 0.0f;
    return inter / ua;
}

int MultiTargetTracker::findTrackIndexById(int id) const
{
    for (size_t i = 0; i < tracks_.size(); ++i)
    {
        if (tracks_[i].id == id)
            return static_cast<int>(i);
    }
    return -1;
}

int MultiTargetTracker::allowedMissedFrames(const TrackState& t) const
{
    // Keep the locked target alive longer to survive short occlusion/fast motion bursts.
    const int lockedBonus = (t.id == lockedTrackId_) ? 8 : 0;
    return maxMissedFrames_ + lockedBonus;
}

void MultiTargetTracker::pruneDeadTracks()
{
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(), [&](const TrackState& t) {
            return t.missed > allowedMissedFrames(t);
            }),
        tracks_.end());
}

int MultiTargetTracker::chooseBestTrack(int screenWidth, int screenHeight) const
{
    if (tracks_.empty())
        return -1;

    const double cx = screenWidth * 0.5;
    const double cy = screenHeight * 0.5;

    int bestIdx = -1;
    double bestScore = std::numeric_limits<double>::max();

    for (size_t i = 0; i < tracks_.size(); ++i)
    {
        const auto& t = tracks_[i];
        if (t.missed > allowedMissedFrames(t))
            continue;

        const double dx = t.pivotX - cx;
        const double dy = t.pivotY - cy;
        const double dist = std::hypot(dx, dy);
        const double hitBonus = std::min(5, t.hits) * 4.0;
        const double missPenalty = t.missed * 50.0;
        const double score = dist + missPenalty - hitBonus;

        if (score < bestScore)
        {
            bestScore = score;
            bestIdx = static_cast<int>(i);
        }
    }

    return bestIdx;
}

void MultiTargetTracker::reset()
{
    tracks_.clear();
    nextId_ = 1;
    lockedTrackId_ = -1;
}

void MultiTargetTracker::update(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classes,
    int screenWidth,
    int screenHeight,
    bool disableHeadshot,
    bool keepCurrentLock)
{
    const auto now = std::chrono::steady_clock::now();

    for (auto& t : tracks_)
        t.observedThisFrame = false;

    if (boxes.size() != classes.size())
    {
        pruneDeadTracks();
        return;
    }

    std::vector<DetectionCandidate> dets;
    dets.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const int cls = classes[i];
        if (disableHeadshot)
        {
            if (cls != config.class_player)
                continue;
        }
        else
        {
            if (cls != config.class_player && cls != config.class_head)
            {
                continue;
            }
        }

        const cv::Rect& b = boxes[i];
        const double yOffset = (cls == config.class_head) ? config.head_y_offset : config.body_y_offset;
        DetectionCandidate d;
        d.box = cv::Rect2f(static_cast<float>(b.x), static_cast<float>(b.y), static_cast<float>(b.width), static_cast<float>(b.height));
        d.classId = cls;
        d.pivotX = b.x + b.width * 0.5;
        d.pivotY = b.y + b.height * yOffset;
        dets.push_back(d);
    }

    if (!disableHeadshot && !dets.empty())
    {
        // If head and player detections refer to the same entity, keep one track candidate
        // (player box for stable identity) but move its pivot to the head point.
        std::vector<size_t> playerIdx;
        playerIdx.reserve(dets.size());
        for (size_t i = 0; i < dets.size(); ++i)
        {
            if (dets[i].classId == config.class_player)
                playerIdx.push_back(i);
        }

        if (!playerIdx.empty())
        {
            std::vector<char> dropHead(dets.size(), 0);
            std::vector<char> playerHasHeadPivot(dets.size(), 0);
            std::vector<double> playerHeadPivotX(dets.size(), 0.0);
            std::vector<double> playerHeadPivotY(dets.size(), 0.0);
            std::vector<double> playerHeadPivotDist(dets.size(), std::numeric_limits<double>::max());

            for (size_t hi = 0; hi < dets.size(); ++hi)
            {
                const auto& h = dets[hi];
                if (h.classId != config.class_head)
                    continue;

                const double headCx = h.box.x + h.box.width * 0.5;
                const double headCy = h.box.y + h.box.height * 0.5;

                size_t bestPlayer = static_cast<size_t>(-1);
                double bestDist = std::numeric_limits<double>::max();

                for (size_t pi : playerIdx)
                {
                    const auto& p = dets[pi].box;
                    const double px1 = p.x - p.width * 0.15;
                    const double px2 = p.x + p.width * 1.15;
                    const double py1 = p.y - p.height * 0.20;
                    const double py2 = p.y + p.height * 0.65;

                    if (!(headCx >= px1 && headCx <= px2 && headCy >= py1 && headCy <= py2))
                        continue;

                    const double pCx = p.x + p.width * 0.5;
                    const double pCy = p.y + p.height * 0.5;
                    const double d = std::hypot(headCx - pCx, headCy - pCy);
                    if (d < bestDist)
                    {
                        bestDist = d;
                        bestPlayer = pi;
                    }
                }

                if (bestPlayer != static_cast<size_t>(-1))
                {
                    dropHead[hi] = 1;
                    if (!playerHasHeadPivot[bestPlayer] || bestDist < playerHeadPivotDist[bestPlayer])
                    {
                        playerHasHeadPivot[bestPlayer] = 1;
                        playerHeadPivotDist[bestPlayer] = bestDist;
                        playerHeadPivotX[bestPlayer] = h.box.x + h.box.width * 0.5;
                        playerHeadPivotY[bestPlayer] = h.box.y + h.box.height * config.head_y_offset;
                    }
                }
            }

            std::vector<DetectionCandidate> filtered;
            filtered.reserve(dets.size());

            for (size_t i = 0; i < dets.size(); ++i)
            {
                if (dropHead[i])
                    continue;

                DetectionCandidate d = dets[i];
                if (d.classId == config.class_player && playerHasHeadPivot[i])
                {
                    d.pivotX = playerHeadPivotX[i];
                    d.pivotY = playerHeadPivotY[i];
                }
                filtered.push_back(d);
            }

            dets.swap(filtered);
        }
    }

    std::vector<int> detAssigned(dets.size(), -1);
    std::vector<int> trackAssigned(tracks_.size(), -1);

    auto computeMatchScore = [&](const TrackState& t, const DetectionCandidate& d, bool relaxedForLocked) -> double
        {
            const bool sameClass = (d.classId == t.classId);
            double classPenalty = 0.0;
            if (!sameClass)
            {
                const bool allowHeadBodySwap =
                    !disableHeadshot &&
                    ((t.classId == config.class_player && d.classId == config.class_head) ||
                     (t.classId == config.class_head && d.classId == config.class_player));
                if (!allowHeadBodySwap)
                    return std::numeric_limits<double>::infinity();

                classPenalty = 0.18;
            }

            const double dt = std::clamp(
                std::chrono::duration<double>(now - t.lastUpdate).count(),
                1e-4, 0.25
            );

            const float predCx = t.box.x + t.box.width * 0.5f + t.velocity.x * static_cast<float>(dt);
            const float predCy = t.box.y + t.box.height * 0.5f + t.velocity.y * static_cast<float>(dt);
            cv::Rect2f predBox(predCx - t.box.width * 0.5f, predCy - t.box.height * 0.5f, t.box.width, t.box.height);

            const double detCx = d.box.x + d.box.width * 0.5;
            const double detCy = d.box.y + d.box.height * 0.5;
            const double dist = std::hypot(detCx - predCx, detCy - predCy);

            const double diag = std::hypot(static_cast<double>(t.box.width), static_cast<double>(t.box.height));
            const double speed = std::hypot(t.velocity.x, t.velocity.y);
            const double baseGate = std::max(24.0, diag * 1.15 + 10.0);
            const double speedGate = speed * dt * (1.8 + t.missed * 0.35);
            const double missGate = t.missed * std::max(14.0, diag * 0.18);
            double maxDist = baseGate + speedGate + missGate;
            if (relaxedForLocked)
                maxDist *= 1.6;

            if (dist > maxDist)
                return std::numeric_limits<double>::infinity();

            const double overlap = iou(predBox, d.box);
            const double missPenalty = t.missed * 0.025;
            const double hitBonus = std::min(6, t.hits) * 0.01;
            return (dist / maxDist) + (1.0 - overlap) * 0.30 + classPenalty + missPenalty - hitBonus;
        };

    auto tryAssignTrack = [&](int trackIndex, bool relaxedForLocked)
        {
            if (trackIndex < 0 || trackIndex >= static_cast<int>(tracks_.size()))
                return;
            if (trackAssigned[trackIndex] != -1)
                return;

            double bestScore = std::numeric_limits<double>::infinity();
            int bestDet = -1;
            const auto& track = tracks_[trackIndex];

            for (size_t di = 0; di < dets.size(); ++di)
            {
                if (detAssigned[di] != -1)
                    continue;

                const double score = computeMatchScore(track, dets[di], relaxedForLocked);
                if (score < bestScore)
                {
                    bestScore = score;
                    bestDet = static_cast<int>(di);
                }
            }

            if (bestDet >= 0)
            {
                trackAssigned[trackIndex] = bestDet;
                detAssigned[bestDet] = trackIndex;
            }
        };

    // Always try to keep the locked track on the same identity first.
    if (lockedTrackId_ != -1)
    {
        const int lockedIdx = findTrackIndexById(lockedTrackId_);
        if (lockedIdx >= 0)
            tryAssignTrack(lockedIdx, true);
    }

    while (true)
    {
        double bestScore = std::numeric_limits<double>::max();
        int bestTi = -1;
        int bestDi = -1;

        for (size_t ti = 0; ti < tracks_.size(); ++ti)
        {
            if (trackAssigned[ti] != -1)
                continue;

            const auto& t = tracks_[ti];

            for (size_t di = 0; di < dets.size(); ++di)
            {
                if (detAssigned[di] != -1)
                    continue;
                const auto& d = dets[di];
                const double score = computeMatchScore(t, d, false);
                if (!std::isfinite(score))
                    continue;

                if (score < bestScore)
                {
                    bestScore = score;
                    bestTi = static_cast<int>(ti);
                    bestDi = static_cast<int>(di);
                }
            }
        }

        if (bestTi < 0 || bestDi < 0)
            break;

        trackAssigned[bestTi] = bestDi;
        detAssigned[bestDi] = bestTi;
    }

    for (size_t ti = 0; ti < tracks_.size(); ++ti)
    {
        auto& t = tracks_[ti];
        const int di = trackAssigned[ti];

        if (di >= 0)
        {
            const auto& d = dets[di];
            const double dt = std::clamp(
                std::chrono::duration<double>(now - t.lastUpdate).count(),
                1e-4, 0.2
            );

            const float oldCx = t.box.x + t.box.width * 0.5f;
            const float oldCy = t.box.y + t.box.height * 0.5f;
            const float newCx = d.box.x + d.box.width * 0.5f;
            const float newCy = d.box.y + d.box.height * 0.5f;
            const cv::Point2f rawVel(
                static_cast<float>((newCx - oldCx) / dt),
                static_cast<float>((newCy - oldCy) / dt)
            );

            cv::Point2f clampedRawVel = rawVel;
            const double rawSpeed = std::hypot(clampedRawVel.x, clampedRawVel.y);
            const double maxReasonableSpeed = std::max(screenWidth, screenHeight) * 3.5;
            if (rawSpeed > maxReasonableSpeed && rawSpeed > 1e-4)
            {
                const float scale = static_cast<float>(maxReasonableSpeed / rawSpeed);
                clampedRawVel *= scale;
            }

            const float blend = (t.id == lockedTrackId_) ? 0.45f : 0.35f;
            t.velocity = t.velocity * (1.0f - blend) + clampedRawVel * blend;
            t.box = d.box;
            t.pivotX = d.pivotX;
            t.pivotY = d.pivotY;
            t.classId = d.classId;
            t.hits += 1;
            t.missed = 0;
            t.observedThisFrame = true;
            t.lastUpdate = now;
        }
        else
        {
            const double dt = std::clamp(
                std::chrono::duration<double>(now - t.lastUpdate).count(),
                0.0, 0.2
            );
            t.box.x += t.velocity.x * static_cast<float>(dt);
            t.box.y += t.velocity.y * static_cast<float>(dt);
            t.pivotX += t.velocity.x * dt;
            t.pivotY += t.velocity.y * dt;
            const float decay = (t.id == lockedTrackId_) ? 0.90f : 0.84f;
            t.velocity *= decay;
            t.missed += 1;
            t.observedThisFrame = false;
            t.lastUpdate = now;
        }
    }

    for (size_t di = 0; di < dets.size(); ++di)
    {
        if (detAssigned[di] != -1)
            continue;

        const auto& d = dets[di];
        TrackState t;
        t.id = nextId_++;
        t.box = d.box;
        t.classId = d.classId;
        t.hits = 1;
        t.missed = 0;
        t.observedThisFrame = true;
        t.pivotX = d.pivotX;
        t.pivotY = d.pivotY;
        t.lastUpdate = now;
        tracks_.push_back(t);
    }

    pruneDeadTracks();

    if (findTrackIndexById(lockedTrackId_) < 0)
        lockedTrackId_ = -1;

    if (!keepCurrentLock)
    {
        const int bestIdx = chooseBestTrack(screenWidth, screenHeight);
        lockedTrackId_ = (bestIdx >= 0) ? tracks_[bestIdx].id : -1;
        return;
    }

    if (lockedTrackId_ == -1)
    {
        const int bestIdx = chooseBestTrack(screenWidth, screenHeight);
        lockedTrackId_ = (bestIdx >= 0) ? tracks_[bestIdx].id : -1;
    }
}

bool MultiTargetTracker::getLockedTarget(LockedTargetInfo& out) const
{
    const int idx = findTrackIndexById(lockedTrackId_);
    if (idx < 0)
        return false;

    const auto& t = tracks_[idx];
    if (t.missed > allowedMissedFrames(t))
        return false;

    out.trackId = t.id;
    out.observedThisFrame = t.observedThisFrame;
    out.missedFrames = t.missed;
    out.target = AimbotTarget(
        static_cast<int>(std::lround(t.box.x)),
        static_cast<int>(std::lround(t.box.y)),
        static_cast<int>(std::lround(t.box.width)),
        static_cast<int>(std::lround(t.box.height)),
        t.classId,
        t.pivotX,
        t.pivotY
    );
    return true;
}

std::vector<TrackDebugInfo> MultiTargetTracker::getDebugTracks() const
{
    std::vector<TrackDebugInfo> out;
    out.reserve(tracks_.size());

    for (const auto& t : tracks_)
    {
        if (t.missed > allowedMissedFrames(t))
            continue;

        TrackDebugInfo d;
        d.trackId = t.id;
        d.classId = t.classId;
        d.box = cv::Rect(
            static_cast<int>(std::lround(t.box.x)),
            static_cast<int>(std::lround(t.box.y)),
            static_cast<int>(std::lround(t.box.width)),
            static_cast<int>(std::lround(t.box.height))
        );
        d.pivotX = t.pivotX;
        d.pivotY = t.pivotY;
        d.observedThisFrame = t.observedThisFrame;
        d.missedFrames = t.missed;
        d.isLocked = (t.id == lockedTrackId_);
        out.push_back(d);
    }

    return out;
}
