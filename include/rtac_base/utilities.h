#ifndef _DEF_RTAC_BASE_UTILITIES_H_
#define _DEF_RTAC_BASE_UTILITIES_H_

#include <deque>
#include <array>
#include <istream>
#include <string>

namespace rtac {

/**
 * Parse a set of numbers from a string when the number is known at compile
 * time.
 */
template <typename T, unsigned int Count>
std::array<T,Count> parse_numbers(std::istream& is, char delimiter = ',')
{
    std::array<T,Count> res;
    std::string token;
    for(unsigned int i = 0; i < Count; i++) {
        if(std::getline(is, token, delimiter)) {
            res[i] = std::stod(token);
        }
    }
    return res;
}

/**
 * Parse a set of numbers from a string. 
 */
template <typename T>
std::deque<T> parse_numbers(std::istream& is, int count, char delimiter = ',')
{
    std::deque<T> res;
    std::string token;
    for(unsigned int i = 0; i < count; i++) {
        if(std::getline(is, token, delimiter)) {
            res.push_back(std::stod(token));
        }
    }
    return res;
}

/**
 * Parse a set of number from a string unitl it is empty.
 */
template <typename T>
std::deque<T> parse_numbers(std::istream& is, char delimiter = ',')
{
    std::deque<T> res;
    std::string token;
    while(std::getline(is, token, delimiter)) {
        res.push_back(std::stod(token));
    }
    return res;
}

}; // namespace rtac

#endif //_DEF_RTAC_BASE_UTILITIES_H_
