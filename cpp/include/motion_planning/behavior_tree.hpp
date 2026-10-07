#pragma once

/// Chapter 9 — behaviour trees: a small engine with sequences, fallbacks, parallels, decorators and a
/// blackboard. Theory: 1_theory/09_behavior_trees.md. Equation numbers below refer to that file.

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace motion_planning::bt {

/// What a node returns when ticked. Section 9.1.
enum class Status { Success, Failure, Running };

std::string to_string(Status s);

/// A value on the blackboard.
using Value = std::variant<bool, int, double, std::string>;

/// Shared memory of a tree: named values that leaves read and write. Section 9.4.
class Blackboard {
  public:
    void set(const std::string& key, const Value& value) { values_[key] = value; }
    bool has(const std::string& key) const { return values_.count(key) > 0; }
    const Value& get(const std::string& key) const;
    /// Numeric value as double (bool and int are converted); throws for strings or missing keys.
    double number(const std::string& key) const;
    const std::map<std::string, Value>& values() const { return values_; }

  private:
    std::map<std::string, Value> values_;
};

/// Base class of all nodes.
class Node {
  public:
    explicit Node(std::string name) : name_(std::move(name)) {}
    virtual ~Node() = default;

    /// Tick the node: run its logic once and return its status. Records the status (see last_status()).
    Status tick(Blackboard& bb);
    /// Stop a running node and reset its internal state (and its children's). Section 9.3.
    virtual void halt();

    const std::string& name() const { return name_; }
    virtual std::string kind() const = 0;
    const std::vector<std::shared_ptr<Node>>& children() const { return children_; }
    /// Status returned by the most recent tick, or nothing if the node has not been ticked since the last
    /// halt.
    bool ticked() const { return ticked_; }
    Status last_status() const { return last_; }
    int tick_count() const { return ticks_; }

  protected:
    virtual Status on_tick(Blackboard& bb) = 0;
    std::vector<std::shared_ptr<Node>> children_;

  private:
    std::string name_;
    bool ticked_ = false;
    Status last_ = Status::Failure;
    int ticks_ = 0;
};

using NodePtr = std::shared_ptr<Node>;

// ---- composites
// ---------------------------------------------------------------------------------------------------

/// Sequence: tick children left to right; fail at the first failure, run at the first running child, succeed
/// if all succeed. Eq. (9.1). With memory, a running child is resumed on the next tick without re-ticking the
/// children before it; without memory (reactive) every tick starts from the first child, and a running child
/// that is no longer reached is halted.
class Sequence : public Node {
  public:
    Sequence(std::string name, std::vector<NodePtr> children, bool memory = false);
    std::string kind() const override { return memory_ ? "SequenceWithMemory" : "Sequence"; }
    void halt() override;

  protected:
    Status on_tick(Blackboard& bb) override;

  private:
    bool memory_;
    std::size_t current_ = 0;
};

/// Fallback (selector): tick children left to right; succeed at the first success, run at the first running
/// child, fail if all fail. Eq. (9.2). Memory works as for Sequence.
class Fallback : public Node {
  public:
    Fallback(std::string name, std::vector<NodePtr> children, bool memory = false);
    std::string kind() const override { return memory_ ? "FallbackWithMemory" : "Fallback"; }
    void halt() override;

  protected:
    Status on_tick(Blackboard& bb) override;

  private:
    bool memory_;
    std::size_t current_ = 0;
};

/// Parallel: tick all children every tick; succeed when at least `success_threshold` succeed, fail when so
/// many have failed that the threshold can no longer be reached, run otherwise. Running children are halted
/// once the parallel decides. Eq. (9.3).
class Parallel : public Node {
  public:
    Parallel(std::string name, std::vector<NodePtr> children, int success_threshold);
    std::string kind() const override { return "Parallel"; }

  protected:
    Status on_tick(Blackboard& bb) override;

  private:
    int threshold_;
};

// ---- decorators
// ---------------------------------------------------------------------------------------------------

/// Inverter: Success <-> Failure, Running unchanged.
class Inverter : public Node {
  public:
    Inverter(std::string name, NodePtr child);
    std::string kind() const override { return "Inverter"; }

  protected:
    Status on_tick(Blackboard& bb) override;
};

/// Retry: re-tick a failing child up to `attempts` times in total (within the same tick if it fails
/// immediately).
class Retry : public Node {
  public:
    Retry(std::string name, NodePtr child, int attempts);
    std::string kind() const override { return "Retry"; }
    void halt() override;

  protected:
    Status on_tick(Blackboard& bb) override;

  private:
    int attempts_, failures_ = 0;
};

/// Repeat: tick a succeeding child `times` times in total; fail as soon as it fails.
class Repeat : public Node {
  public:
    Repeat(std::string name, NodePtr child, int times);
    std::string kind() const override { return "Repeat"; }
    void halt() override;

  protected:
    Status on_tick(Blackboard& bb) override;

  private:
    int times_, successes_ = 0;
};

/// Timeout: fail (and halt the child) if the child is still running after `max_ticks` ticks.
class Timeout : public Node {
  public:
    Timeout(std::string name, NodePtr child, int max_ticks);
    std::string kind() const override { return "Timeout"; }
    void halt() override;

  protected:
    Status on_tick(Blackboard& bb) override;

  private:
    int max_ticks_, running_for_ = 0;
};

/// ForceSuccess / ForceFailure: report success (failure) whatever the child returns, unless it is running.
class Force : public Node {
  public:
    Force(std::string name, NodePtr child, Status forced);
    std::string kind() const override { return forced_ == Status::Success ? "ForceSuccess" : "ForceFailure"; }

  protected:
    Status on_tick(Blackboard& bb) override;

  private:
    Status forced_;
};

// ---- leaves
// -------------------------------------------------------------------------------------------------------

/// Condition: Success if the predicate holds, Failure otherwise; never Running.
class Condition : public Node {
  public:
    Condition(std::string name, std::function<bool(Blackboard&)> predicate);
    std::string kind() const override { return "Condition"; }

  protected:
    Status on_tick(Blackboard& bb) override;

  private:
    std::function<bool(Blackboard&)> predicate_;
};

/// Action: runs a user function that returns a status. An optional callback runs when the action is halted
/// while running (to stop a motor, cancel a plan, reset a counter).
class Action : public Node {
  public:
    Action(std::string name, std::function<Status(Blackboard&)> run, std::function<void()> on_halt = {});
    std::string kind() const override { return "Action"; }
    void halt() override;

  protected:
    Status on_tick(Blackboard& bb) override;

  private:
    std::function<Status(Blackboard&)> run_;
    std::function<void()> on_halt_;
};

/// Condition comparing a numeric blackboard entry with a constant: op is one of < <= > >= == !=.
NodePtr compare(const std::string& name, const std::string& key, const std::string& op, double value);

/// Action that writes a value and succeeds.
NodePtr set_value(const std::string& name, const std::string& key, const Value& value);

/// Action that is Running for `ticks` ticks and then succeeds (restarts after a halt).
NodePtr wait(const std::string& name, int ticks);

// ---- tree
// ---------------------------------------------------------------------------------------------------------

/// A tree with its blackboard; tick() ticks the root once.
class Tree {
  public:
    explicit Tree(NodePtr root) : root_(std::move(root)) {}
    Status tick();
    /// Tick until the root returns Success or Failure, or `max_ticks` ticks have passed. Returns the last
    /// status.
    Status run(int max_ticks);
    void halt() { root_->halt(); }
    Blackboard& blackboard() { return bb_; }
    const NodePtr& root() const { return root_; }
    int ticks() const { return ticks_; }
    /// The tree as indented text with each node's last status.
    std::string to_string() const;

  private:
    NodePtr root_;
    Blackboard bb_;
    int ticks_ = 0;
};

}  // namespace motion_planning::bt
