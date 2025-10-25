(define (domain move_with_persons_domain)
  (:requirements :strips :typing)

  (:types
    robot
    waypoint
    object
    person
  )

  (:predicates
    ;; navigation
    (robot_at ?r - robot ?w - waypoint)
    (connected ?from - waypoint ?to - waypoint)

    ;; pick-and-place
    (object_at ?o - object ?w - waypoint)
    (holding ?r - robot ?o - object)
    (handempty ?r - robot)

    ;; persons
    (person_at ?p - person ?w - waypoint)
    (person_holding ?p - person ?o - object)
  )

  ;; move between waypoints
  (:action move
    :parameters (?r - robot ?from - waypoint ?to - waypoint)
    :precondition (and
      (robot_at ?r ?from)
      (connected ?from ?to)
    )
    :effect (and
      (not (robot_at ?r ?from))
      (robot_at ?r ?to)
    )
  )

  ;; take from person
  (:action take
    :parameters (?r - robot ?p - person ?o - object ?w - waypoint)
    :precondition (and
      (robot_at ?r ?w)
      (person_at ?p ?w)
      (person_holding ?p ?o)
      (handempty ?r)
    )
    :effect (and
      (holding ?r ?o)
      (not (person_holding ?p ?o))
      (not (handempty ?r))
    )
  )

  ;; give to person
  (:action give
    :parameters (?r - robot ?p - person ?o - object ?w - waypoint)
    :precondition (and
      (robot_at ?r ?w)
      (person_at ?p ?w)
      (holding ?r ?o)
    )
    :effect (and
      (person_holding ?p ?o)
      (handempty ?r)
      (not (holding ?r ?o))
    )
  )
)
