#ifndef _DEF_RTAC_BASE_ASYNC_ASYNC_FUNCTION_H_
#define _DEF_RTAC_BASE_ASYNC_ASYNC_FUNCTION_H_

#include <type_traits>
#include <functional>
#include <future>

namespace rtac {

/**
 * This type allows for a unified return type deduction, choosing between
 * std::result_of or std::invoke_result depending on compiler option
 *
 * TODO : only std::result_of is used for now, implement for std::invoke_result
 * -> This is actually difficult to doin a portable way because there is no
 * reliable way to check in code for c++ version for gcc and MVSC.
 */
template <class F, class... Args>
struct ResultOf
{
    using type = typename std::result_of<F&&(Args&&...)>::type;
};

/**
 * This is a base type for the async::AsyncFunction<T> type.
 * 
 * Its may purpose is to provide an abstract interface to async::AsyncFunction
 * to be able to store them in a single container.
 */
struct AsyncFunctionBase
{
    using Ptr = std::unique_ptr<AsyncFunctionBase>;

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

    using Ptr = std::unique_ptr<AsyncFunction<R>>;
    using result_type = R;

    protected:

    std::function<R(void)> function_;
    std::promise<R>        promise_;

    public:

    AsyncFunction() = delete;
    AsyncFunction(const AsyncFunction<R>&) = delete;
    AsyncFunction<R>& operator=(const AsyncFunction<R>&) = delete;

    AsyncFunction(std::function<R(void)>&& f) : 
        function_(std::move(f))
    {}

    AsyncFunction(AsyncFunction<R>&& f) :
        function_(std::move(f.function_)),
        promise_ (std::move(f.promise_))
    {}
    AsyncFunction<R>& operator=(AsyncFunction<R>&& f) {
        function_ = std::move(f.function_);
        promise_  = std::move(f.promise_);
    }

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

    using Ptr = std::unique_ptr<AsyncFunction<void>>;
    using result_type = void;

    protected:

    std::function<void(void)> function_;
    std::promise<void>        promise_;

    public:

    AsyncFunction() = delete;
    AsyncFunction(const AsyncFunction<void>&) = delete;
    AsyncFunction<void>& operator=(const AsyncFunction<void>&) = delete;

    AsyncFunction(std::function<void(void)>&& f) : 
        function_(std::move(f))
    {}

    AsyncFunction(AsyncFunction<void>&& f) :
        function_(std::move(f.function_)),
        promise_ (std::move(f.promise_))
    {}
    AsyncFunction<void>& operator=(AsyncFunction<void>&& f) {
        function_ = std::move(f.function_);
        promise_  = std::move(f.promise_);
    }

    std::future<void> future() { return promise_.get_future(); }
    void execute() { function_(); promise_.set_value();        }
};

/**
 * This is a special specialization of AsyncFunctionBase. The only goal is to
 * provide a synchronization primitive and does nothing otherwise
 */
class EmptyAsyncFunction : public AsyncFunctionBase
{
    public:

    using Ptr = std::unique_ptr<EmptyAsyncFunction>;
    using result_type = void;

    protected:

    std::promise<void> promise_;

    public:

    EmptyAsyncFunction() {}

    std::future<void> future() { return promise_.get_future(); }
    void execute()             { promise_.set_value();         }
};

/**
 * Creates a new AsyncFunction from a callable and its argument. It is
 * compatible with function pointers and lambda expressions with capture
 * (untested on callable object but should work).
 */
template <class F, class... Args> inline
auto async_bind(F f, Args&&... args)
{
    using R = typename ResultOf<F,Args...>::type;
    return std::make_unique<AsyncFunction<R>>(std::move(
        std::function<R()>(std::bind(f, std::forward<Args>(args)...))));
}

/**
 * Creates a new AsyncFunction from a method pointer, a class instance, and its
 * arguments.
 *
 * The two parameter packs allow for implicit conversion to happen.
 */
template <class R, class C, class...Args1, class... Args2> inline typename
AsyncFunction<R>::Ptr async_bind(R(C::*f)(Args1...), C* c, Args2&&... args)
{
    return std::make_unique<AsyncFunction<R>>(std::move(
        std::function<R()>(std::bind(f, c, std::forward<Args2>(args)...))));
}

/**
 * Creates a new AsyncFunction from a method pointer, a class instance, and its
 * arguments (const version)
 *
 * The two parameter packs allow for implicit conversion to happen.
 */
template <class R, class C, class...Args1, class... Args2> inline typename
AsyncFunction<R>::Ptr async_bind(R(C::*f)(Args1...) const, const C* c, Args2&&... args)
{
    return std::make_unique<AsyncFunction<R>>(std::move(
        std::function<R()>(std::bind(f, c, std::forward<Args2>(args)...))));
}

/**
 * Creates an empty synchronization primitive which does nothing, but can be
 * synchronized upon.
 */
inline EmptyAsyncFunction::Ptr empty_async()
{
    return std::make_unique<EmptyAsyncFunction>();
}

} //namespace rtac

#endif //_DEF_RTAC_BASE_ASYNC_ASYNC_FUNCTION_H_
