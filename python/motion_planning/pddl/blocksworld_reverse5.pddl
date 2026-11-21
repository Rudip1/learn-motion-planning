; Reverse a tower of five blocks: a on b on c on d on e  ->  e on d on c on b on a.
(define (problem reverse5)
  (:domain blocksworld)
  (:objects a b c d e)
  (:init (on a b) (on b c) (on c d) (on d e) (ontable e) (clear a) (handempty))
  (:goal (and (on e d) (on d c) (on c b) (on b a))))
