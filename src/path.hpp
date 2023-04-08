#ifndef _has_path_hpp_
#define _has_path_hpp_

#include <exception>
#include <list>
#include <memory>

namespace utl {

  template <class T>
  class path : public std::enable_shared_from_this<path<T>> {
  public: // member type aliases
    using pos_t = T;
    using ptr_t = std::shared_ptr<path>;

  private: // data fields
    pos_t _position;
    ptr_t _previous;
    uint _lives;
    std::list<ptr_t> _next;

  private: // constructors are private: must use factory instead

    path() = default;
    path(const path&) = delete;
    path& operator=(const path&) = delete;

  public: // methods

    // path factory

    static ptr_t create(
      const pos_t position,
      const uint lives = 1
    )
    {
      auto raw_ptr = new path();
      raw_ptr->_position = position;
      raw_ptr->_lives = lives;
      return ptr_t(raw_ptr);
    }

    // helper function to retrieve a shared_ptr of this

    ptr_t get_ptr()
    {return this->shared_from_this();}

    // walk towards the given position

    ptr_t walk(
      const pos_t pos,
      const uint lives = 0
    )
    {
      auto s = create(pos);
      s->_previous = get_ptr();
      s->_live = lives == 0 ? get_lives() : lives;
      _next.push_back(s);
      return s;
    }

    // data access

    const auto get_lives() const {return _lives;}
    const auto get_next() const {return _next;}
    const auto get_position() const {return _position;}
    const auto get_previous() const {return _previous;}

    // convert path into a list of positions

    auto get_position_list() {
      std::list<pos_t> pos_list;
      for (auto p = get_ptr(); p != nullptr; p = p->get_previous()) {
        pos_list.push_front(p->get_position());
      }
      return pos_list;
    }

    // drop the current path

    void drop() {
      if (!_next.empty()) {
        throw std::runtime_error("paths can only be dropped from their last elements");
      }

      // if this path is connected to a previous path
      if (_previous) {
        // then remove this from the previous path
        _previous->_next.remove(get_ptr());
        // if previous is not connected to any other path, drop it too
        if (_previous->_next.empty()) {
          _previous->drop();
        }
      }

      // remove reference to previous, so it can be deleted
      _previous = nullptr;
    }
  };

} // utl

#endif // _has_path_hpp_