(define (problem move_two_boxes_with_persons)
  (:domain move_with_persons_domain)

  (:objects
    r1 - robot
    wp0 wp1 wp2 wp3 wp4 - waypoint
    box1 box2 - object
    person1 person2 person3 person4 - person
  )

  (:init
    (robot_at r1 wp0)
    (person_at person1 wp1)
    (person_holding person1 box1)
    (person_at person2 wp2)
    (person_holding person2 box2)
    (person_at person3 wp3)
    (person_at person4 wp4)
    (connected wp0 wp1)
    (connected wp1 wp3)
    (connected wp3 wp2)
    (connected wp2 wp4)
    (connected wp4 wp0)
    (handempty r1)
  )

  (:goal
    (and
      (person_holding person3 box1)
      (person_holding person4 box2)
      (robot_at r1 wp0)
      (handempty r1)
    )
  )
)
