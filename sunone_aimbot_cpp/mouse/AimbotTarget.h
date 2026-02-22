#ifndef AIMBOTTARGET_H
#define AIMBOTTARGET_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

class AimbotTarget
{
public:
    AimbotTarget();
    int x, y, w, h;
    int classId;

    double pivotX;
    double pivotY;

    AimbotTarget(int x, int y, int w, int h, int classId, double pivotX = 0.0, double pivotY = 0.0);
};

AimbotTarget* sortTargets(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classes,
    int screenWidth,
    int screenHeight,
    bool disableHeadshot
);

struct LockedTargetInfo
{
    int trackId = -1;
    bool observedThisFrame = false;
    int missedFrames = 0;
    AimbotTarget target;
};

struct TrackDebugInfo
{
    int trackId = -1;
    int classId = -1;
    cv::Rect box;
    double pivotX = 0.0;
    double pivotY = 0.0;
    bool observedThisFrame = false;
    int missedFrames = 0;
    bool isLocked = false;
};

class MultiTargetTracker
{
public:
    void reset();
    void update(
        const std::vector<cv::Rect>& boxes,
        const std::vector<int>& classes,
        int screenWidth,
        int screenHeight,
        bool disableHeadshot,
        bool keepCurrentLock
    );
    bool getLockedTarget(LockedTargetInfo& out) const;
    int getLockedTrackId() const { return lockedTrackId_; }
    std::vector<TrackDebugInfo> getDebugTracks() const;

private:
    struct TrackState
    {
        int id = -1;
        cv::Rect2f box;
        cv::Point2f velocity = { 0.0f, 0.0f };
        int classId = -1;
        int hits = 0;
        int missed = 0;
        bool observedThisFrame = false;
        double pivotX = 0.0;
        double pivotY = 0.0;
        std::chrono::steady_clock::time_point lastUpdate;
    };

    struct DetectionCandidate
    {
        cv::Rect2f box;
        int classId = -1;
        double pivotX = 0.0;
        double pivotY = 0.0;
    };

    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);
    int findTrackIndexById(int id) const;
    int chooseBestTrack(int screenWidth, int screenHeight) const;
    int allowedMissedFrames(const TrackState& t) const;
    void pruneDeadTracks();

    std::vector<TrackState> tracks_;
    int nextId_ = 1;
    int lockedTrackId_ = -1;
    int maxMissedFrames_ = 6;
};

#endif // AIMBOTTARGET_H
