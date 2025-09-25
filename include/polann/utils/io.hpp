#pragma once

template <typename T>
concept Streamable = requires(std::ostream &os, const T &t) {
    { os << t } -> std::same_as<std::ostream &>;
};

template <Streamable T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "[";
    for (size_t i = 0; i < v.size(); i++)
    {
        os << v[i];
        if (i + 1 < v.size())
            os << ", ";
    }
    os << "]";
    return os;
}
