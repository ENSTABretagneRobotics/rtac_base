#ifndef _DEF_RTAC_BASE_ASYNC_ASYNC_FUNCTION_H_
#define _DEF_RTAC_BASE_ASYNC_ASYNC_FUNCTION_H_

#include <functional>
#include <future>

namespace rtac { namespace async {

/**
 * This is a base type for the async::AsyncFunction<T> type.
 * 
 * Its may purpose is to provide an abstract interface to async::AsyncFunction
 * to be able to store them in a single container.
 */
struct AsyncFunctionBase
{
    using Ptr = std::shared_ptr<AsyncFunctionBase>;

    void operator()() { this->execute(); }

    virtual void execute() = 0;
};

/**
 * This stores a std::function to be executed at a later time and a
 * std::promise to get the result.
 *
 * The std::promise object acts as a synchonization primitive to wait for the
 * result.
 */
template <class R>
class AsyncFunction : public AsyncFunctionBase
{
    public:

    using Ptr = std::shared_ptr<AsyncFunction<R>>;
    using result_type = R;

    protected:

    std::function<R(void)> function_;
    std::promise<R>        promise_;

    public:

    AsyncFunction(std::function<R(void)>&& f) : 
        function_(std::move(f))
    {}

    std::future<R> future() { return promise_.get_future(); }
    void execute() { promise_.set_value(function_());   }
};

/**
 * This is a specialization of AsyncFunction for void return type. It is
 * identical except for the execute method which has to be modified because we
 * cannot call the std::promise::set_value method directly on the result of a
 * void function.
 *
 * The std::promise object acts as a synchonization primitive to wait for the
 * result.
 */
template <>
class AsyncFunction<void> : public AsyncFunctionBase
{
    public:

    using Ptr = std::shared_ptr<AsyncFunction<void>>;
    using result_type = void;

    protected:

    std::function<void(void)> function_;
    std::promise<void>        promise_;

    public:

    AsyncFunction(std::function<void(void)>&& f) :
        function_(std::move(f))
    {}

    std::future<void> future() { return promise_.get_future(); }
    void execute() { function_(); promise_.set_value();        }
};

template <class R> inline typename 
AsyncFunction<R>::Ptr make_async_function(std::function<R(void)>&& f)
{
    return std::make_shared<AsyncFunction<R>>(std::forward<std::function<R(void)>>(f));
}

inline typename 
AsyncFunction<void>::Ptr make_async_function(std::function<void(void)>&& f)
{
    return std::make_shared<AsyncFunction<void>>(std::forward<std::function<void(void)>>(f));
}

} //namespace async
} //namespace rtac

#endif //_DEF_RTAC_BASE_ASYNC_ASYNC_FUNCTION_H_
