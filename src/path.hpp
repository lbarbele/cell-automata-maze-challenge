#ifndef _has_path_hpp_
#define _has_path_hpp_

// ! beware the use of "position" here!

#include <exception>
#include <list>
#include <memory>

template <class T>
class path : public std::enable_shared_from_this<path<T>> {
public:
  using position_t = T;
  using ptr_t = std::shared_ptr<path<position_t>>;

private:
  position_t _position;
  ptr_t _previous;
  std::list<ptr_t> _next;

  path() = default;

  path(const path&) = delete;
  path& operator=(const path&) = delete;

public:

  static ptr_t create(
    const position_t position,
    const ptr_t previous = nullptr
  )
  {
    auto s = ptr_t(new path());
    s->_position = position;
    s->_previous = previous;
    return s;
  }

  ptr_t get_ptr()
  {return this->shared_from_this();}

  ptr_t walk(const position_t pos)
  {
    auto s = create(pos, get_ptr());
    _next.push_back(s);
    return s;
  }

  const auto get_next() const {return _next;}
  const auto get_position() const {return _position;}
  const auto get_previous() const {return _previous;}

  auto get_position_list() {
    std::list<position_t> pos_list;
    for (auto p = get_ptr(); p != nullptr; p = p->get_previous()) {
      pos_list.push_front(p->get_position());
    }
    return pos_list;
  }

  void drop() {
    if (!_next.empty()) {
      throw std::runtime_error("can not drop a path if _next is not empty");
    }

    if (_previous) {
      _previous->_next.remove(get_ptr());
      if (_previous->_next.empty()) {
        _previous->drop();
      }
    }

    _previous = nullptr;
  }
};

#endif // _has_path_hpp_