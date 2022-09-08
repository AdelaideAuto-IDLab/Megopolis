#pragma once

#include <stdio.h> 
#include <utility>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <exception>
#include <cerrno>
#include "../../include/rapidjson/document.h"
#include "../../include/rapidjson/filereadstream.h"
#include "utility.h"

// Implement some primive type parsers for json values

float parse_float(const rapidjson::Value * dom, const char * name) {
    if (!dom->HasMember(name)) throw std::invalid_argument(std::string("Missing ") + name);
    if (!(*dom)[name].IsFloat()) throw std::invalid_argument(std::string("Field not float: ") + name);

    return (*dom)[name].GetFloat();
}

std::string parse_string(const rapidjson::Value * dom, const char * name) {
    if (!dom->HasMember(name)) throw std::invalid_argument(std::string("Missing ") + name);
    if (!((*dom)[name].IsString())) throw std::invalid_argument(std::string("Field not string: ") + name);
    
    return (*dom)[name].GetString();
}

template<size_t N>
std::array<float, N> parse_array(const rapidjson::Value * dom, const char * name) {
    if (!dom->HasMember(name)) throw std::invalid_argument(std::string("Missing ") + name);
    if (!(*dom)[name].IsArray()) throw std::invalid_argument(std::string("Field is not an array: ") + name);

    const auto& array = (*dom)[name].GetArray();
    if (array.Size() != N) throw std::invalid_argument(std::string("Array is incorrect size: ") + name);

    std::array<float, N> output;

    for (size_t i = 0; i < N; i++) {
        if (!array[i].IsFloat()) {
            char buffer[200];
            int n = sprintf(buffer, "%i th value in array not float %s", int(i), name);

            if (n < 0) throw std::runtime_error("Failed to write array error"); 

            throw std::invalid_argument(buffer);
        } 

        output[i] = array[i].GetFloat();
    }

    return output;
}   

size_t parse_uint(const rapidjson::Value * dom, const char * name) {
    if (!dom->HasMember(name)) throw std::invalid_argument(std::string("Missing ") + name);
    if (!(*dom)[name].IsUint()) throw std::invalid_argument(std::string("Field not uint: ") + name);
    
    return (*dom)[name].GetUint();
}

std::pair<std::array<std::vector<float>, 2>, bool> parse_gain_angles(const char * path, char next_row) {
    std::ifstream file(path, std::ifstream::in);
    std::vector<float> phi;
    std::vector<float> theta;
    std::string buffer;
    std::string::size_type offset;

    while (file.good()) {
        if (!getline(file, buffer, next_row)) {
            break;
        }

        try {
            float phi_angle = std::stof(buffer, &offset);
            float theta_angle = std::stof(buffer.substr(offset + 1));
            phi.push_back(phi_angle);
            theta.push_back(theta_angle);
        }
        catch (...) {
            std::array<std::vector<float>, 2> array = { phi, theta };
            return std::make_pair(array, false);
        }
    }   

    std::array<std::vector<float>, 2> array = { phi, theta };
    return std::make_pair(array, true);
}

// Annoying to template the return type to we template the input type
void parse_value(const rapidjson::Value * dom, float &into);
void parse_value(const rapidjson::Value * dom, std::string &into);
void parse_value(const rapidjson::Value * dom, size_t &into);
void parse_value(const rapidjson::Value * dom, int &into);
template <typename T> 
void parse_value(const rapidjson::Value * dom, std::vector<T> &into);
template <size_t N, typename T> 
void parse_value(const rapidjson::Value * dom, std::array<T, N> &into);

// Overloading the parse_value function 

void parse_value(const rapidjson::Value * dom, bool &into) {
    if (!dom->IsBool()) throw std::invalid_argument(std::string("Failed to parse float"));

    into = dom->GetBool();
}

void parse_value(const rapidjson::Value * dom, float &into) {
    if (!dom->IsFloat()) throw std::invalid_argument(std::string("Failed to parse float"));

    into = dom->GetFloat();
}

void parse_value(const rapidjson::Value * dom, std::string &into) {
    if (!dom->IsString()) throw std::invalid_argument(std::string("Failed to parse string"));

    into = dom->GetString();
}

void parse_value(const rapidjson::Value * dom, size_t &into) {
    if (dom->IsUint()) {
        into = dom->GetUint();
        return; 
    }
    else if (dom->IsInt()) {
        int value = dom->GetInt(); 

        if (value < 0) {
            throw std::invalid_argument(std::string("Got negative value for UInt"));
        }

        into = (size_t) value; 
        return;
    }

    throw std::invalid_argument(std::string("Failed to parse uint: "));
}

void parse_value(const rapidjson::Value * dom, int &into) {
    if (!dom->IsInt()) throw std::invalid_argument(std::string("Failed to parse int"));

    into = dom->GetInt();
}

template <typename T> 
void parse_value(const rapidjson::Value * dom, std::vector<T> &into) {
    if (!dom->IsArray()) throw std::invalid_argument(std::string("Failed to parse array"));
    const auto& array = dom->GetArray();
    size_t size = array.Size();

    for (size_t i = 0; i < size; i++) {
        T value;
        parse_value(&array[i], value);
        into.emplace_back(std::move(value));
    }   
}

template<size_t N, typename T> 
void parse_value(const rapidjson::Value * dom, std::array<T, N> &into) {
    if (!dom->IsArray()) throw std::invalid_argument(std::string("Failed to parse array"));
    const auto& array = dom->GetArray();
    size_t size = array.Size();

    if (size != N) throw std::invalid_argument(std::string("Failed to parse array, incorrect size"));

    for (size_t i = 0; i < N; i++) {
        T value;
        parse_value(&array[i], value);
        into[i] = std::move(value);
    }   
}

template<typename T> 
void parse_field(const rapidjson::Value * dom, const char * name, T &into) {
    if (!dom->HasMember(name)) throw std::invalid_argument(std::string("Missing ") + name);
    
    parse_value(&(*dom)[name], into);
}

template<typename T>
void parse_field(const rapidjson::Value * dom, const char * name, Option<T> &into) {
    if (!dom->HasMember(name)) {
        into = Option<T>();
        return;
    }

    parse_value(&(*dom)[name], into.unwrap_ref());
    into.valid = true;
}

template<typename T> 
void parse_field(const rapidjson::Value * dom, const char * name, T &into, T def) {
    if (!dom->HasMember(name)) {
        into = def;
        return;
    }

    parse_value(&(*dom)[name], into);
}

rapidjson::Document load_config(const char * path) {
    FILE * f = fopen(path, "r");

    if (f == nullptr) {
        std::string error("Failed to open \"");
        error.append(path);
        error.append("\": ");
        error.append(std::strerror(errno));

        throw std::invalid_argument(error);
    }

    // @Polish: This is bad if yo the file is ever bigger than 10000 chars...
    char readbuf[10000];
    rapidjson::FileReadStream is(f, readbuf, sizeof(readbuf));

    rapidjson::Document d;
    d.ParseStream(is);

    fclose(f);

    return d;
}