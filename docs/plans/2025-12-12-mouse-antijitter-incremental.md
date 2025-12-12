# Mouse Anti-Jitter Incremental Refactoring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add anti-jitter smoothing to the mouse module without breaking existing functionality, using incremental changes that can be tested independently.

**Architecture:** Layer smoothing ON TOP of existing code. Each task adds ONE feature that can be toggled via config. Velocity smoothing first (biggest impact), then deadzone, then detection delay. No double-filtering, no breaking changes.

**Tech Stack:** C++17, Windows API, OpenCV (for types only)

---

## Task 0: Create Branch and Clean Up Broken Changes

**Goal:** Create feature branch and revert/clean the broken anti-jitter changes from config and UI while keeping the UI structure.

**Files:**
- Git operations
- Modify: `sunone_aimbot_cpp/config/config.h`
- Modify: `sunone_aimbot_cpp/config/config.cpp`
- Modify: `sunone_aimbot_cpp/overlay/draw_mouse.cpp`

### Step 1: Create feature branch

```powershell
cd "C:\Users\therouxe\source\repos\MacroMan5\macroman_aimbot_cpp"
git checkout -b feat/refactor-mouse
```

### Step 2: Clean config.h - Replace broken params with correct ones

Open `sunone_aimbot_cpp/config/config.h`. Find the Anti-jitter section (around line 55-63) and **REPLACE**:

**BEFORE (broken):**
```cpp
    // Anti-jitter smoothing
    float position_smoothing_alpha;      // 0.4 - EMA on detected position
    float velocity_smoothing_alpha;      // 0.3 - EMA on calculated velocity
    int mouse_deadzone;                  // 3 - pixels ignored
    float stationary_velocity_threshold; // 50.0 - stationary target threshold (px/s)
    float wind_disable_distance;         // 10.0 - disable wind mouse when close
```

**AFTER (correct):**
```cpp
    // Anti-jitter smoothing
    float velocity_smoothing;            // 0.5 - EMA factor (0=none, 0.9=max smoothing)
    int pixel_deadzone;                  // 2 - ignore when within N pixels of center
    float stationary_threshold;          // 30.0 - velocity below this = no prediction (px/s)
```

**Note:** Removed `position_smoothing_alpha` (caused double filtering) and `wind_disable_distance` (not needed for MVP).

### Step 3: Clean config.cpp - Update defaults and load/save

Open `sunone_aimbot_cpp/config/config.cpp`.

**In constructor/defaults section** (around line 96-100), **REPLACE**:

```cpp
    // Anti-jitter (REPLACE old values)
    velocity_smoothing = 0.5f;
    pixel_deadzone = 2;
    stationary_threshold = 30.0f;
```

**In loadConfig function** (around line 341-345), **REPLACE**:

```cpp
    // Anti-jitter
    velocity_smoothing = (float)get_double("velocity_smoothing", 0.5);
    pixel_deadzone = get_long("pixel_deadzone", 2);
    stationary_threshold = (float)get_double("stationary_threshold", 30.0);
```

**In saveConfig function** (around line 530-534), **REPLACE**:

```cpp
        << "velocity_smoothing = " << velocity_smoothing << "\n"
        << "pixel_deadzone = " << pixel_deadzone << "\n"
        << "stationary_threshold = " << stationary_threshold << "\n"
```

### Step 4: Clean draw_mouse.cpp - Fix UI to match new params

Open `sunone_aimbot_cpp/overlay/draw_mouse.cpp`.

**At the top** (around line 36-40), **REPLACE** the prev_ variables:

```cpp
// Anti-jitter smoothing
float prev_velocity_smoothing = config.velocity_smoothing;
int prev_pixel_deadzone = config.pixel_deadzone;
float prev_stationary_threshold = config.stationary_threshold;
```

**In the Anti-Jitter UI section** (around line 145-180), **REPLACE** the entire block:

```cpp
    ImGui::SeparatorText("Anti-Jitter Smoothing");

    ImGui::SliderFloat("Velocity Smoothing", &config.velocity_smoothing, 0.0f, 0.9f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("0 = no smoothing (raw), higher = smoother but more lag. Start at 0.5");

    ImGui::SliderInt("Pixel Deadzone", &config.pixel_deadzone, 0, 10);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Stop moving when within N pixels of target center. Prevents micro-oscillation.");

    ImGui::SliderFloat("Stationary Threshold", &config.stationary_threshold, 0.0f, 100.0f, "%.0f px/s");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Velocity below this = target is stationary, disable prediction.");

    if (ImGui::Button("Reset Anti-Jitter"))
    {
        config.velocity_smoothing = 0.5f;
        config.pixel_deadzone = 2;
        config.stationary_threshold = 30.0f;
        config.saveConfig();
    }
```

**At the bottom** (around line 697-720), **REPLACE** the change detection:

```cpp
    // Anti-jitter changes
    if (prev_velocity_smoothing != config.velocity_smoothing ||
        prev_pixel_deadzone != config.pixel_deadzone ||
        prev_stationary_threshold != config.stationary_threshold)
    {
        prev_velocity_smoothing = config.velocity_smoothing;
        prev_pixel_deadzone = config.pixel_deadzone;
        prev_stationary_threshold = config.stationary_threshold;
        config.saveConfig();
    }
```

### Step 5: Build to verify config/UI compiles

```powershell
cd "C:\Users\therouxe\source\repos\MacroMan5\macroman_aimbot_cpp"
msbuild sunone_aimbot_cpp.sln /p:Configuration=DML /p:Platform=x64 /v:minimal
```

Expected: Build succeeds (mouse.cpp doesn't use these params yet).

### Step 6: Test UI appears correctly

Run `x64\DML\ai.exe`, press HOME, go to Mouse tab. Verify:
- "Anti-Jitter Smoothing" section exists
- 3 sliders: Velocity Smoothing, Pixel Deadzone, Stationary Threshold
- Reset button works

### Step 7: Commit clean state

```bash
git add -A
git commit -m "chore: clean up broken anti-jitter params, prepare for correct implementation"
```

---

## Task 1: Implement Velocity EMA Smoothing

**Goal:** Add actual velocity smoothing to mouse.cpp using the new config param.

**Files:**
- Modify: `sunone_aimbot_cpp/mouse/mouse.h:40-44`
- Modify: `sunone_aimbot_cpp/mouse/mouse.cpp:359-362`

### Step 1: Add smoothed velocity members to header

Open `sunone_aimbot_cpp/mouse/mouse.h`. After line 44 (`std::atomic<bool> target_detected{ false };`), add:

```cpp
    // Anti-jitter: smoothed velocity state
    double smoothed_vx = 0.0;
    double smoothed_vy = 0.0;
```

### Step 2: Implement velocity smoothing in moveMousePivot

Open `sunone_aimbot_cpp/mouse/mouse.cpp`. Find `moveMousePivot` function, lines 359-362.

**REPLACE:**
```cpp
    double vx = std::clamp((pivotX - prev_x) / dt, -20000.0, 20000.0);
    double vy = std::clamp((pivotY - prev_y) / dt, -20000.0, 20000.0);
    prev_x = pivotX; prev_y = pivotY;
    prev_velocity_x = vx;  prev_velocity_y = vy;
```

**WITH:**
```cpp
    // Calculate raw velocity
    double raw_vx = std::clamp((pivotX - prev_x) / dt, -20000.0, 20000.0);
    double raw_vy = std::clamp((pivotY - prev_y) / dt, -20000.0, 20000.0);

    // Apply EMA smoothing: lower alpha = more smoothing
    // velocity_smoothing: 0 = no smooth (alpha=1), 0.9 = max smooth (alpha=0.1)
    double alpha = 1.0 - config.velocity_smoothing;
    double vx = alpha * raw_vx + (1.0 - alpha) * smoothed_vx;
    double vy = alpha * raw_vy + (1.0 - alpha) * smoothed_vy;

    // Store smoothed values for next frame
    smoothed_vx = vx;
    smoothed_vy = vy;
    prev_x = pivotX;
    prev_y = pivotY;
    prev_velocity_x = vx;
    prev_velocity_y = vy;
```

### Step 3: Reset smoothed velocity in resetPrediction

In `mouse.cpp`, find `resetPrediction()` (around line 480). Add after `prev_velocity_y = 0;`:

```cpp
    smoothed_vx = 0;
    smoothed_vy = 0;
```

### Step 4: Also reset in first-frame block of moveMousePivot

In `moveMousePivot`, in the first-frame block (around line 346-352), add after `prev_velocity_x = prev_velocity_y = 0.0;`:

```cpp
        smoothed_vx = smoothed_vy = 0.0;
```

### Step 5: Build and test

```powershell
msbuild sunone_aimbot_cpp.sln /p:Configuration=DML /p:Platform=x64 /v:minimal
```

**Test:**
- Run ai.exe
- Set Velocity Smoothing to 0 → should behave like before (raw)
- Set to 0.5 → should be noticeably smoother on stationary target
- Set to 0.8 → very smooth but laggy

### Step 6: Commit

```bash
git add sunone_aimbot_cpp/mouse/mouse.h sunone_aimbot_cpp/mouse/mouse.cpp
git commit -m "feat(mouse): implement velocity EMA smoothing for anti-jitter"
```

---

## Task 2: Implement Pixel-Based Deadzone

**Goal:** Stop moving when target is within N pixels of screen center.

**Files:**
- Modify: `sunone_aimbot_cpp/mouse/mouse.cpp`

### Step 1: Add deadzone check after velocity calculation

Open `sunone_aimbot_cpp/mouse/mouse.cpp`. In `moveMousePivot`, after the velocity smoothing code (after `prev_velocity_y = vy;`), add:

```cpp
    // Pixel deadzone: stop if target within N pixels of center
    if (config.pixel_deadzone > 0)
    {
        double offset_x = pivotX - center_x;
        double offset_y = pivotY - center_y;
        double offset_dist = std::hypot(offset_x, offset_y);

        if (offset_dist < config.pixel_deadzone)
        {
            return;  // Within deadzone, don't move
        }
    }
```

### Step 2: Build and test

```powershell
msbuild sunone_aimbot_cpp.sln /p:Configuration=DML /p:Platform=x64 /v:minimal
```

**Test:**
- Set Pixel Deadzone to 0 → normal behavior
- Set to 5 → when very close to target center, mouse stops completely
- Set to 10 → larger "lock-on" zone

### Step 3: Commit

```bash
git add sunone_aimbot_cpp/mouse/mouse.cpp
git commit -m "feat(mouse): implement pixel-based deadzone"
```

---

## Task 3: Implement Dynamic Detection Delay

**Goal:** Use actual inference time instead of hardcoded 0.002.

**Files:**
- Modify: `sunone_aimbot_cpp/mouse/mouse.cpp:364-365`

### Step 1: Replace hardcoded delay

In `moveMousePivot`, find the prediction lines (after deadzone check):

**REPLACE:**
```cpp
    double predX = pivotX + vx * prediction_interval + vx * 0.002;
    double predY = pivotY + vy * prediction_interval + vy * 0.002;
```

**WITH:**
```cpp
    // Get actual detection delay from inference timing
    double detection_delay = 0.016;  // Default 16ms fallback
    if (config.backend == "DML" && dml_detector)
    {
        detection_delay = dml_detector->lastInferenceTimeDML.count() / 1000.0;
    }
#ifdef USE_CUDA
    else if (config.backend == "TRT")
    {
        detection_delay = trt_detector.lastInferenceTime.count() / 1000.0;
    }
#endif
    detection_delay = std::clamp(detection_delay, 0.001, 0.1);  // 1-100ms range

    double predX = pivotX + vx * (prediction_interval + detection_delay);
    double predY = pivotY + vy * (prediction_interval + detection_delay);
```

### Step 2: Build and test

```powershell
msbuild sunone_aimbot_cpp.sln /p:Configuration=DML /p:Platform=x64 /v:minimal
```

**Test:** Prediction should now adapt to actual inference speed shown in stats.

### Step 3: Commit

```bash
git add sunone_aimbot_cpp/mouse/mouse.cpp
git commit -m "feat(mouse): use dynamic detection delay for prediction"
```

---

## Task 4: Implement Stationary Target Detection

**Goal:** Disable prediction when target velocity is very low (prevents jitter amplification).

**Files:**
- Modify: `sunone_aimbot_cpp/mouse/mouse.cpp`

### Step 1: Add stationary check and conditional prediction

In `moveMousePivot`, **REPLACE** the prediction block (the detection_delay + predX/predY code from Task 3) with:

```cpp
    // Check if target is stationary
    double velocity_magnitude = std::hypot(vx, vy);
    bool is_stationary = (config.stationary_threshold > 0) &&
                         (velocity_magnitude < config.stationary_threshold);

    double predX, predY;
    if (is_stationary)
    {
        // Target stationary - aim directly without prediction
        predX = pivotX;
        predY = pivotY;
    }
    else
    {
        // Target moving - apply prediction with detection delay
        double detection_delay = 0.016;
        if (config.backend == "DML" && dml_detector)
        {
            detection_delay = dml_detector->lastInferenceTimeDML.count() / 1000.0;
        }
#ifdef USE_CUDA
        else if (config.backend == "TRT")
        {
            detection_delay = trt_detector.lastInferenceTime.count() / 1000.0;
        }
#endif
        detection_delay = std::clamp(detection_delay, 0.001, 0.1);

        predX = pivotX + vx * (prediction_interval + detection_delay);
        predY = pivotY + vy * (prediction_interval + detection_delay);
    }
```

### Step 2: Build and test

```powershell
msbuild sunone_aimbot_cpp.sln /p:Configuration=DML /p:Platform=x64 /v:minimal
```

**Test:**
- Stationary target: Should aim smoothly without oscillation
- Moving target: Should still predict correctly
- Set threshold to 0: Disables feature, always predicts

### Step 3: Commit

```bash
git add sunone_aimbot_cpp/mouse/mouse.cpp
git commit -m "feat(mouse): add stationary target detection to disable prediction when appropriate"
```

---

## Task 5: Final Testing and Cleanup

**Goal:** Verify everything works together, clean up any issues.

### Step 1: Full test matrix

| Scenario | Velocity Smoothing | Pixel Deadzone | Stationary Threshold | Expected |
|----------|-------------------|----------------|---------------------|----------|
| Raw mode | 0 | 0 | 0 | Original behavior |
| Smooth only | 0.5 | 0 | 0 | Smoother velocity |
| Full anti-jitter | 0.5 | 2 | 30 | Smooth, no micro-oscillation |
| Max smooth | 0.8 | 5 | 50 | Very stable but laggy |

### Step 2: Test each scenario

Run ai.exe and test with each config combination above.

### Step 3: Verify config persistence

- Change values in UI
- Restart application
- Verify values are restored

### Step 4: Final commit

```bash
git add -A
git commit -m "test: verify anti-jitter implementation complete"
```

### Step 5: Push branch

```bash
git push -u origin feat/refactor-mouse
```

---

## Summary

| Task | Feature | Lines Changed | Risk |
|------|---------|---------------|------|
| 0 | Clean broken params | ~30 | LOW |
| 1 | Velocity smoothing | ~15 | LOW |
| 2 | Pixel deadzone | ~10 | LOW |
| 3 | Dynamic delay | ~15 | LOW |
| 4 | Stationary detection | ~20 | LOW |
| 5 | Testing | 0 | NONE |

### Default Values (Recommended Starting Point)

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `velocity_smoothing` | 0.5 | 0.0-0.9 | 0=raw, higher=smoother |
| `pixel_deadzone` | 2 | 0-10 | pixels from center |
| `stationary_threshold` | 30.0 | 0-100 | px/s, 0=disabled |

### Key Differences from Broken Implementation

| Aspect | Broken | Fixed |
|--------|--------|-------|
| Position filtering | EMA on position | NONE (not needed) |
| Velocity filtering | Double EMA | Single EMA |
| Deadzone metric | Mouse counts | Screen pixels |
| Filter order | EMA before target check | Target check first |
| Detection delay | Missing | Dynamic from inference |
