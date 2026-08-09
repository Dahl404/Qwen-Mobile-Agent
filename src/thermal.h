/* thermal.h — silent device governor. A background thread watches CPU
 * temperature + load and caps the engine's worker pool when hot (and pins
 * to the efficiency cores at critical), restoring full speed when cool.
 * No output anywhere — the user and the model never see it.
 */
#ifndef THERMAL_H
#define THERMAL_H

void thermal_start(void);
void thermal_stop(void);

#endif /* THERMAL_H */
