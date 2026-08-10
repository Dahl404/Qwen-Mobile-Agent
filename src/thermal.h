/* thermal.h — silent device governor. A background thread watches CPU
 * temperature + load and caps the engine's worker pool when hot (and pins
 * to the efficiency cores at critical), restoring full speed when cool.
 * Level changes are QUEUED and drained by the app at a quiet point
 * (before the prompt) — never printed mid-generation where they would
 * interleave with the model output.
 */
#ifndef THERMAL_H
#define THERMAL_H

void thermal_start(void);
void thermal_stop(void);
/* print any queued level-change message to stderr (no-op when none).
   Call from the main loop right before the prompt, never mid-output. */
void thermal_drain(void);

#endif /* THERMAL_H */
