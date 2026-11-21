; Swap two boxes between two stands in different rooms, using a third stand in the corridor as a buffer.
; Places: room1 - corridor - room2 (a line). Optimal plan: 13 actions (see 1_theory/10_task_planning.md).
(define (problem swap)
  (:domain pick-place)
  (:objects room1 corridor room2 - place  s1 s2 s3 - stand  red blue - box)
  (:init (robot-at corridor)
         (connected room1 corridor) (connected corridor room1) (connected corridor room2) (connected room2 corridor)
         (stand-at s1 room1) (stand-at s2 room2) (stand-at s3 corridor)
         (on red s1) (on blue s2) (occupied s1) (occupied s2) (gripper-free))
  (:goal (and (on red s2) (on blue s1))))
