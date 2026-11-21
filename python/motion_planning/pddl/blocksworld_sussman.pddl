; The Sussman anomaly: C is on A, B is on the table; build the tower A on B on C. Achieving the two goals one after
; the other in either order undoes the first, so a planner must interleave them. Optimal plan: 6 actions.
(define (problem sussman)
  (:domain blocksworld)
  (:objects a b c)
  (:init (on c a) (ontable a) (ontable b) (clear c) (clear b) (handempty))
  (:goal (and (on a b) (on b c))))
