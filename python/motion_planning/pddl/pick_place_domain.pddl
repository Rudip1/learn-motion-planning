; A mobile manipulator with one suction gripper moves boxes between stands at different places (the task layer
; of the pick-and-place robot in vehicle-manipulator-task-priority). Typed, with a negative precondition: a box can
; be put only on a free stand.
(define (domain pick-place)
  (:requirements :strips :typing :negative-preconditions)
  (:types place stand box)
  (:predicates (robot-at ?p - place) (stand-at ?s - stand ?p - place) (on ?b - box ?s - stand)
               (occupied ?s - stand) (holding ?b - box) (gripper-free) (connected ?from ?to - place))
  (:action drive
    :parameters (?from ?to - place)
    :precondition (and (robot-at ?from) (connected ?from ?to))
    :effect (and (robot-at ?to) (not (robot-at ?from))))
  (:action pick
    :parameters (?b - box ?s - stand ?p - place)
    :precondition (and (robot-at ?p) (stand-at ?s ?p) (on ?b ?s) (gripper-free))
    :effect (and (holding ?b) (not (on ?b ?s)) (not (occupied ?s)) (not (gripper-free))))
  (:action place
    :parameters (?b - box ?s - stand ?p - place)
    :precondition (and (robot-at ?p) (stand-at ?s ?p) (holding ?b) (not (occupied ?s)))
    :effect (and (on ?b ?s) (occupied ?s) (gripper-free) (not (holding ?b)))))
