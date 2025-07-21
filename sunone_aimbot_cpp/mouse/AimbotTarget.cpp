#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#define NOMINMAX
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <limits>
#include <vector>
#include <optional>
#include <opencv2/core/types.hpp>

#include "sunone_aimbot_cpp.h"
#include "AimbotTarget.h"
#include "config.h"

// El constructor no requiere cambios, ya es eficiente.
AimbotTarget::AimbotTarget(int x_, int y_, int w_, int h_, int cls, double px, double py)
    : x(x_), y(y_), w(w_), h(h_), classId(cls), pivotX(px), pivotY(py)
{
}

// Devuelve std::optional para evitar alocaciones de memoria en el heap (new/delete).
std::optional<AimbotTarget> sortTargets(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classes,
    int screenWidth,
    int screenHeight,
    bool disableHeadshot)
{
    if (boxes.empty()) {
        return std::nullopt; // Retorno inmediato si no hay objetivos detectados.
    }

    // Cálculos del centro de la pantalla. Se usa double para mayor precisión.
    const double centerX = static_cast<double>(screenWidth) / 2.0;
    const double centerY = static_cast<double>(screenHeight) / 2.0;

    // *** MODIFICADO: Radio del FOV para la selección de objetivos ***
    // Este es el radio en píxeles (al cuadrado para evitar usar sqrt()) dentro del cual el aimbot buscará objetivos.
    // Ahora se basa en el valor `fovX` de la configuración para que sea personalizable.
    // NOTA: Esto define el ÁREA DE BÚSQUEDA, no el comportamiento de "snap" o "lock-on".
    const double aimbotFovRadiusSq = static_cast<double>(config.fovX) * static_cast<double>(config.fovX);

    double minDistanceSq = std::numeric_limits<double>::max();
    int nearestIdx = -1;

    // --- Búsqueda de la cabeza (si está habilitada) ---
    if (!disableHeadshot)
    {
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            if (classes[i] == config.class_head)
            {
                const auto& box = boxes[i];
                const double targetX = static_cast<double>(box.x) + static_cast<double>(box.width) / 2.0;
                const double targetY = static_cast<double>(box.y) + static_cast<double>(box.height) * config.head_y_offset;
                
                // Calcular distancia al cuadrado para eficiencia.
                const double dx = targetX - centerX;
                const double dy = targetY - centerY;
                const double distanceSq = dx * dx + dy * dy;

                if (distanceSq < minDistanceSq)
                {
                    minDistanceSq = distanceSq;
                    nearestIdx = static_cast<int>(i);
                }
            }
        }
    }

    // --- Búsqueda del cuerpo (si no se encontró cabeza o los disparos a la cabeza están deshabilitados) ---
    if (nearestIdx == -1)
    {
        minDistanceSq = std::numeric_limits<double>::max(); // Reiniciar distancia para la búsqueda del cuerpo
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            const int classId = classes[i];
            bool isTargetClass = (classId == config.class_player ||
                                  classId == config.class_bot ||
                                  (config.shooting_range_targets && (classId == config.class_hideout_target_human || classId == config.class_hideout_target_balls)) ||
                                  (!config.ignore_third_person && classId == config.class_third_person));

            if (isTargetClass)
            {
                const auto& box = boxes[i];
                const double targetX = static_cast<double>(box.x) + static_cast<double>(box.width) / 2.0;
                const double targetY = static_cast<double>(box.y) + static_cast<double>(box.height) * config.body_y_offset;

                const double dx = targetX - centerX;
                const double dy = targetY - centerY;
                const double distanceSq = dx * dx + dy * dy;

                if (distanceSq < minDistanceSq)
                {
                    minDistanceSq = distanceSq;
                    nearestIdx = static_cast<int>(i);
                }
            }
        }
    }

    // Si después de todas las búsquedas no encontramos un objetivo válido o está fuera del FOV, retornamos.
    if (nearestIdx == -1 || minDistanceSq > aimbotFovRadiusSq)
    {
        return std::nullopt;
    }
   
    // --- Construcción del objeto AimbotTarget final ---
    const auto& finalBox = boxes[nearestIdx];
    const int finalClass = classes[nearestIdx];
    
    double pivotX = static_cast<double>(finalBox.x) + static_cast<double>(finalBox.width) / 2.0;
    double pivotY;

    if (finalClass == config.class_head) {
        pivotY = static_cast<double>(finalBox.y) + static_cast<double>(finalBox.height) * config.head_y_offset;
    } else {
        pivotY = static_cast<double>(finalBox.y) + static_cast<double>(finalBox.height) * config.body_y_offset;
    }
    
    // Devolvemos el objetivo por valor. El compilador aplicará RVO (Return Value Optimization),
    // evitando copias y alocaciones innecesarias.
    return AimbotTarget(finalBox.x, finalBox.y, finalBox.width, finalBox.height, finalClass, pivotX, pivotY);
} // AimbotTarget.cpp