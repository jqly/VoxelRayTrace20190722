#ifndef JIANGQILEI_UTIL_H
#define JIANGQILEI_UTIL_H

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>
#include <chrono>
#include "graphics_math.h"

namespace jql
{
inline std::string format(const std::string &fmt)
{
        return fmt;
}

template <typename First, typename... Others>
std::string format(const std::string &fmt, First first, Others... others)
{
        auto is_op_delim = [&fmt](int idx) {
                if (fmt[idx] == '{')
                        return true;
                return false;
        };

        auto is_ed_delim = [&fmt](int idx) {
                if (fmt[idx] == '}')
                        return true;
                return false;
        };

        auto is_delim_escape = [&fmt](int idx) {
                if (idx + 1 >= fmt.size())
                        return false;
                if ((fmt[idx] == '{' && fmt[idx + 1] == '{') ||
                    (fmt[idx] == '}' && fmt[idx + 1] == '}'))
                        return true;
                return false;
        };

        std::string first_part, format_part;
        int state = 0, first_part_ed = 0;
        for (int idx = 0; idx < fmt.size(); ++idx) {
                if (is_delim_escape(idx)) {
                        if (state == 0)
                                first_part += fmt[idx + 1];
                        else if (state == 1)
                                format_part += fmt[idx + 1];
                        idx++;
                }
                else if (is_op_delim(idx)) {
                        if (state == 0)
                                state = 1;
                        else {
                                std::cerr << "Delimiter mismatch.\n";
                                exit(1);
                        }
                }
                else if (is_ed_delim(idx)) {
                        if (state == 1) {
                                state = 2;
                                first_part_ed = idx + 1;
                        }
                        else {
                                std::cerr << "Delimiter mismatch.\n";
                                exit(1);
                        }
                        break;
                }
                else {
                        if (state == 0)
                                first_part += fmt[idx];
                        else if (state == 1)
                                format_part += fmt[idx];
                }
        }
        if (state != 2) {
                std::cerr << "Delimiter mismatch.\n";
                exit(1);
        }
        auto remaining_part =
                fmt.substr(first_part_ed, fmt.size() - first_part_ed);
        std::stringstream ss;

        if (std::is_floating_point<First>::value || jql::is_mat<First>::value ||
            std::is_same<jql::Quat, First>::value) {
                // Setting precision of a floating point number.
                // Changing width of the integral part: Unsupported.
                auto pos = format_part.find('.');
                if (pos != std::string::npos && pos < format_part.size() - 1) {
                        auto curr_prec = std::stoi(format_part.substr(pos + 1));
                        auto prev_prec = ss.precision(curr_prec);
                        ss << first;
                        ss.precision(prev_prec);
                }
                else
                        ss << first;
        }
        else
                ss << first;

        return first_part + ss.str() + format(remaining_part, others...);
}

template <typename... Args>
void print(const std::string &fmt, Args... args)
{
        std::cout << format(fmt, args...);
}

template <typename... Args>
void print(std::ostream &out, const std::string &fmt, Args... args)
{
        out << format(fmt, args...);
}

inline std::string read_file(const std::string &path)
{
        std::ifstream fin(path, std::ios_base::binary);

        assert(!fin.fail());

        fin.ignore(std::numeric_limits<std::streamsize>::max());
        auto size = fin.gcount();
        fin.clear();

        fin.seekg(0, std::ios_base::beg);
        auto source = std::unique_ptr<char>(new char[size]);
        fin.read(source.get(), size);

        return std::string(source.get(),
                           static_cast<std::string::size_type>(size));
}

// No trailling dirsep.
inline std::string get_file_base_dir(const std::string &filepath)
{
        auto probe = filepath.find_last_of("/\\");
        if (probe == std::string::npos)
                return "";
        return filepath.substr(0, probe);
}

inline std::string get_file_extension(const std::string &filename)
{
        auto found = filename.find_last_of(".");
        if (found == std::string::npos) {
                std::cerr << "Wrong input file.\n";
                exit(1);
        }
        return filename.substr(found);
}

} // namespace jql

#endif
