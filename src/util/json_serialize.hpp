#pragma once

#include <cstddef>  
#include <sstream>

namespace json_se {
    template<typename WRITER>
    void comma(WRITER &writer) {
        writer << ",";
    }

    template<typename WRITER>
    void open_brace(WRITER &writer) {
        writer << "{";
    }

    template<typename WRITER>
    void close_brace(WRITER &writer) {
        writer << "}";
    }
    
    template<typename WRITER>
    void open_bracket(WRITER &writer) {
        writer << "[";
    }

    template<typename WRITER>
    void close_bracket(WRITER &writer) {
        writer << "]";
    }

    template<typename WRITER>
    void newline(WRITER &writer) {
        writer << "\n";
    }

    template<typename WRITER, typename T>
    void serialize_value(WRITER &writer, T &value) {
        writer << value;
    }

    template<typename WRITER, typename T>
    void serialize_field(WRITER &writer, const char * name, T &value) {
        writer << "\"" << name << "\":";
        serialize_value(writer, value);
    }

    template<typename WRITER>
    void serialize_field_name(WRITER &writer, const char * name) {
        writer << "\"" << name << "\":";
    }

    template<typename WRITER, typename T>
    void serialize_array(WRITER &writer, const T * values, size_t amount) {
        open_bracket(writer);

        for (size_t i = 1; i < amount; i++) {
            serialize_value(writer, values[i - 1]);
            comma(writer);
        }

        if (amount > 0) {
            serialize_value(writer, values[amount - 1]);
        }

        close_bracket(writer);
    }
};