#include "motion_planning/behavior_tree.hpp"

#include <sstream>
#include <stdexcept>

namespace motion_planning::bt {

std::string to_string(Status s) {
    switch (s) {
        case Status::Success:
            return "SUCCESS";
        case Status::Failure:
            return "FAILURE";
        case Status::Running:
            return "RUNNING";
    }
    return "?";
}

const Value& Blackboard::get(const std::string& key) const {
    const auto it = values_.find(key);
    if (it == values_.end()) throw std::out_of_range("no blackboard entry '" + key + "'");
    return it->second;
}

double Blackboard::number(const std::string& key) const {
    const Value& v = get(key);
    if (const auto* b = std::get_if<bool>(&v)) return *b ? 1.0 : 0.0;
    if (const auto* i = std::get_if<int>(&v)) return *i;
    if (const auto* d = std::get_if<double>(&v)) return *d;
    throw std::invalid_argument("blackboard entry '" + key + "' is not a number");
}

Status Node::tick(Blackboard& bb) {
    ++ticks_;
    last_ = on_tick(bb);
    ticked_ = true;
    return last_;
}

void Node::halt() {
    for (const NodePtr& c : children_) c->halt();
    if (ticked_ && last_ == Status::Running) ticked_ = false;  // shown as idle until ticked again
}

namespace {
void halt_from(const std::vector<NodePtr>& children, std::size_t first) {
    for (std::size_t j = first; j < children.size(); ++j) children[j]->halt();
}
void require_children(const std::vector<NodePtr>& c, const std::string& name) {
    if (c.empty()) throw std::invalid_argument("composite '" + name + "' needs children");
    for (const NodePtr& n : c)
        if (!n) throw std::invalid_argument("composite '" + name + "' has a null child");
}
}  // namespace

Sequence::Sequence(std::string name, std::vector<NodePtr> children, bool memory)
    : Node(std::move(name)), memory_(memory) {
    require_children(children, this->name());
    children_ = std::move(children);
}

Status Sequence::on_tick(Blackboard& bb) {
    // eq. (9.1): the first child that does not succeed decides
    for (std::size_t i = memory_ ? current_ : 0; i < children_.size(); ++i) {
        const Status s = children_[i]->tick(bb);
        if (s == Status::Success) continue;
        halt_from(children_, i + 1);  // children further right may still be running from an earlier tick
        current_ = s == Status::Running ? i : 0;
        return s;
    }
    current_ = 0;
    return Status::Success;
}

void Sequence::halt() {
    current_ = 0;
    Node::halt();
}

Fallback::Fallback(std::string name, std::vector<NodePtr> children, bool memory)
    : Node(std::move(name)), memory_(memory) {
    require_children(children, this->name());
    children_ = std::move(children);
}

Status Fallback::on_tick(Blackboard& bb) {
    // eq. (9.2): the first child that does not fail decides
    for (std::size_t i = memory_ ? current_ : 0; i < children_.size(); ++i) {
        const Status s = children_[i]->tick(bb);
        if (s == Status::Failure) continue;
        halt_from(children_, i + 1);
        current_ = s == Status::Running ? i : 0;
        return s;
    }
    current_ = 0;
    return Status::Failure;
}

void Fallback::halt() {
    current_ = 0;
    Node::halt();
}

Parallel::Parallel(std::string name, std::vector<NodePtr> children, int success_threshold)
    : Node(std::move(name)), threshold_(success_threshold) {
    require_children(children, this->name());
    if (success_threshold < 1 || success_threshold > static_cast<int>(children.size()))
        throw std::invalid_argument("parallel threshold must be between 1 and the number of children");
    children_ = std::move(children);
}

Status Parallel::on_tick(Blackboard& bb) {
    int successes = 0, failures = 0;
    for (const NodePtr& c : children_) {
        const Status s = c->tick(bb);
        successes += s == Status::Success;
        failures += s == Status::Failure;
    }
    const int n = static_cast<int>(children_.size());
    Status result = Status::Running;  // eq. (9.3)
    if (successes >= threshold_)
        result = Status::Success;
    else if (failures > n - threshold_)
        result = Status::Failure;
    if (result != Status::Running) halt_from(children_, 0);
    return result;
}

namespace {
NodePtr checked(NodePtr child) {
    if (!child) throw std::invalid_argument("decorator needs a child");
    return child;
}
}  // namespace

Inverter::Inverter(std::string name, NodePtr child) : Node(std::move(name)) {
    children_ = {checked(std::move(child))};
}

Status Inverter::on_tick(Blackboard& bb) {
    const Status s = children_[0]->tick(bb);
    if (s == Status::Running) return s;
    return s == Status::Success ? Status::Failure : Status::Success;
}

Retry::Retry(std::string name, NodePtr child, int attempts) : Node(std::move(name)), attempts_(attempts) {
    if (attempts < 1) throw std::invalid_argument("retry needs at least one attempt");
    children_ = {checked(std::move(child))};
}

Status Retry::on_tick(Blackboard& bb) {
    while (true) {
        const Status s = children_[0]->tick(bb);
        if (s != Status::Failure) {
            if (s == Status::Success) failures_ = 0;
            return s;
        }
        if (++failures_ >= attempts_) {
            failures_ = 0;
            return Status::Failure;
        }
        children_[0]->halt();  // start the next attempt from scratch
    }
}

void Retry::halt() {
    failures_ = 0;
    Node::halt();
}

Repeat::Repeat(std::string name, NodePtr child, int times) : Node(std::move(name)), times_(times) {
    if (times < 1) throw std::invalid_argument("repeat needs at least one repetition");
    children_ = {checked(std::move(child))};
}

Status Repeat::on_tick(Blackboard& bb) {
    while (true) {
        const Status s = children_[0]->tick(bb);
        if (s == Status::Running) return s;
        if (s == Status::Failure) {
            successes_ = 0;
            return s;
        }
        if (++successes_ >= times_) {
            successes_ = 0;
            return Status::Success;
        }
        children_[0]->halt();
    }
}

void Repeat::halt() {
    successes_ = 0;
    Node::halt();
}

Timeout::Timeout(std::string name, NodePtr child, int max_ticks)
    : Node(std::move(name)), max_ticks_(max_ticks) {
    if (max_ticks < 1) throw std::invalid_argument("timeout needs at least one tick");
    children_ = {checked(std::move(child))};
}

Status Timeout::on_tick(Blackboard& bb) {
    const Status s = children_[0]->tick(bb);
    if (s != Status::Running) {
        running_for_ = 0;
        return s;
    }
    if (++running_for_ >= max_ticks_) {
        children_[0]->halt();
        running_for_ = 0;
        return Status::Failure;
    }
    return Status::Running;
}

void Timeout::halt() {
    running_for_ = 0;
    Node::halt();
}

Force::Force(std::string name, NodePtr child, Status forced) : Node(std::move(name)), forced_(forced) {
    if (forced == Status::Running) throw std::invalid_argument("can only force success or failure");
    children_ = {checked(std::move(child))};
}

Status Force::on_tick(Blackboard& bb) {
    const Status s = children_[0]->tick(bb);
    return s == Status::Running ? s : forced_;
}

Condition::Condition(std::string name, std::function<bool(Blackboard&)> predicate)
    : Node(std::move(name)), predicate_(std::move(predicate)) {
    if (!predicate_) throw std::invalid_argument("condition needs a predicate");
}

Status Condition::on_tick(Blackboard& bb) { return predicate_(bb) ? Status::Success : Status::Failure; }

Action::Action(std::string name, std::function<Status(Blackboard&)> run, std::function<void()> on_halt)
    : Node(std::move(name)), run_(std::move(run)), on_halt_(std::move(on_halt)) {
    if (!run_) throw std::invalid_argument("action needs a function");
}

Status Action::on_tick(Blackboard& bb) { return run_(bb); }

void Action::halt() {
    if (ticked() && last_status() == Status::Running && on_halt_) on_halt_();
    Node::halt();
}

NodePtr compare(const std::string& name, const std::string& key, const std::string& op, double value) {
    std::function<bool(double, double)> f;
    if (op == "<")
        f = [](double a, double b) { return a < b; };
    else if (op == "<=")
        f = [](double a, double b) { return a <= b; };
    else if (op == ">")
        f = [](double a, double b) { return a > b; };
    else if (op == ">=")
        f = [](double a, double b) { return a >= b; };
    else if (op == "==")
        f = [](double a, double b) { return a == b; };
    else if (op == "!=")
        f = [](double a, double b) { return a != b; };
    else
        throw std::invalid_argument("unknown comparison '" + op + "'");
    return std::make_shared<Condition>(name,
                                       [key, f, value](Blackboard& bb) { return f(bb.number(key), value); });
}

NodePtr set_value(const std::string& name, const std::string& key, const Value& value) {
    return std::make_shared<Action>(name, [key, value](Blackboard& bb) {
        bb.set(key, value);
        return Status::Success;
    });
}

NodePtr wait(const std::string& name, int ticks) {
    auto count = std::make_shared<int>(0);
    return std::make_shared<Action>(
        name,
        [count, ticks](Blackboard&) {
            if (++*count >= ticks) {
                *count = 0;
                return Status::Success;
            }
            return Status::Running;
        },
        [count]() { *count = 0; });
}

Status Tree::tick() {
    ++ticks_;
    return root_->tick(bb_);
}

Status Tree::run(int max_ticks) {
    Status s = Status::Running;
    for (int k = 0; k < max_ticks && s == Status::Running; ++k) s = tick();
    return s;
}

namespace {
void print(std::ostringstream& out, const Node& n, int depth) {
    out << std::string(2 * depth, ' ') << n.kind() << " '" << n.name() << "'";
    if (n.ticked()) out << "  [" << to_string(n.last_status()) << "]";
    out << '\n';
    for (const NodePtr& c : n.children()) print(out, *c, depth + 1);
}
}  // namespace

std::string Tree::to_string() const {
    std::ostringstream out;
    print(out, *root_, 0);
    return out.str();
}

}  // namespace motion_planning::bt
