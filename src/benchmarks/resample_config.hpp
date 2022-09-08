#pragma once

#include "../util/parse_json.hpp"
#include <string>
#include <array>

// Implement json parsing for resample testing configuration

struct ResampleParticles
{
    size_t input_size;
    size_t output_size;
};

void parse_value(const rapidjson::Value *dom, ResampleParticles &method)
{
    // Value is [input_size, output_size]
    if (dom->IsArray())
    {
        std::array<size_t, 2> values;
        parse_value(dom, values);

        method.input_size = values[0];
        method.output_size = values[1];
    }
    // value is just size
    else
    {
        size_t value;
        parse_value(dom, value);
        method.input_size = value;
        method.output_size = value;
    }
}

// Select iter mode
enum class IterMode
{
    Dynamic,
    PreCompute,
    Constant,
    Default = Dynamic
};

void parse_value(const rapidjson::Value *dom, IterMode &method)
{
    std::string value;
    parse_value(dom, value);

    if (value == std::string("Dynamic"))
    {
        method = IterMode::Dynamic;
    }
    else if (value == std::string("PreCompute"))
    {
        method = IterMode::PreCompute;
    }
    else
    {
        method = IterMode::Constant;
    }
}

enum class ResampleType
{
    Megopolis,
    MegopolisAligned,
    SegMegopolis,
    Metropolis,
    MetropolisC1,
    MetropolisC2,
    Multinomial,
    Systematic,
    NicelySystematic,
    SortSystematic,
    Stratified
};

// Output name of resample tpe
const char *rtype_to_str(const ResampleType &t)
{
    switch (t)
    {
    case ResampleType::Megopolis:
        return "Megopolis";
    case ResampleType::MegopolisAligned:
        return "MegopolisAligned";
    case ResampleType::Metropolis:
        return "Metropolis";
    case ResampleType::MetropolisC1:
        return "MetropolisC1";
    case ResampleType::SegMegopolis:
        return "SegMegopolis";
    case ResampleType::MetropolisC2:
        return "MetropolisC2";
    case ResampleType::Multinomial:
        return "Multinomial";
    case ResampleType::Systematic:
        return "Systematic";
    case ResampleType::NicelySystematic:
        return "NicelySystematic";
    case ResampleType::SortSystematic:
        return "SortSystematic";
    case ResampleType::Stratified:
        return "Stratified";
    default:
        return "Unknown";
    }
}

// Resample Method configuration
struct ResampleMethod
{
    ResampleType method;
    size_t iters;
    size_t segment_size;
    IterMode mode;
    float iter_ratio = 1.0;

    // Convert method to string for logging purposes
    std::string to_string()
    {
        std::string name(rtype_to_str(method));

        if (mode == IterMode::Constant)
        {
            name.append("_");
            name.append(std::to_string(iters));
        }

        if (method == ResampleType::MetropolisC1)
        {
            name.append("_");
            name.append(std::to_string(segment_size));
        }
        else if (method == ResampleType::SegMegopolis)
        {
            name.append("_");
            name.append(std::to_string(segment_size));
        }
        else if (method == ResampleType::MetropolisC2)
        {
            name.append("_");
            name.append(std::to_string(segment_size));
        }

        return name;
    }
};

// Parse the resample method configuration
void parse_value(const rapidjson::Value *dom, ResampleMethod &method)
{
    std::string value;
    parse_field(dom, "type", value);

    if (value == std::string("Megopolis"))
    {
        method.method = ResampleType::Megopolis;
        parse_field(dom, "mode", method.mode);
        if (method.mode == IterMode::Constant)
        {
            parse_field(dom, "iters", method.iters);
        }
        else
        {
            parse_field(dom, "estimate_ratio", method.iter_ratio, 1.0f);
        }
    }
    else if (value == std::string("MegopolisAligned"))
    {
        method.method = ResampleType::MegopolisAligned;
        parse_field(dom, "mode", method.mode);
        if (method.mode == IterMode::Constant)
        {
            parse_field(dom, "iters", method.iters);
        }
        else
        {
            parse_field(dom, "estimate_ratio", method.iter_ratio, 1.0f);
        }
    }
    else if (value == std::string("SegMegopolis"))
    {
        method.method = ResampleType::SegMegopolis;
        parse_field(dom, "segment_size", method.segment_size);
        parse_field(dom, "mode", method.mode);
        if (method.mode == IterMode::Constant)
        {
            parse_field(dom, "iters", method.iters);
        }
        else
        {
            parse_field(dom, "estimate_ratio", method.iter_ratio, 1.0f);
        }
    }
    else if (value == std::string("MetropolisC1"))
    {
        method.method = ResampleType::MetropolisC1;
        parse_field(dom, "segment_size", method.segment_size);
        parse_field(dom, "mode", method.mode);
        if (method.mode == IterMode::Constant)
        {
            parse_field(dom, "iters", method.iters);
        }
        else
        {
            parse_field(dom, "estimate_ratio", method.iter_ratio, 1.0f);
        }
    }
    else if (value == std::string("MetropolisC2"))
    {
        method.method = ResampleType::MetropolisC2;
        parse_field(dom, "segment_size", method.segment_size);
        parse_field(dom, "mode", method.mode);
        if (method.mode == IterMode::Constant)
        {
            parse_field(dom, "iters", method.iters);
        }
        else
        {
            parse_field(dom, "estimate_ratio", method.iter_ratio, 1.0f);
        }
    }
    else if (value == std::string("Metropolis"))
    {
        method.method = ResampleType::Metropolis;
        parse_field(dom, "mode", method.mode);
        if (method.mode == IterMode::Constant)
        {
            parse_field(dom, "iters", method.iters);
        }
        else
        {
            parse_field(dom, "estimate_ratio", method.iter_ratio, 1.0f);
        }
    }
    else if (value == std::string("Multinomial"))
    {
        method.method = ResampleType::Multinomial;
    }
    else if (value == std::string("Systematic"))
    {
        method.method = ResampleType::Systematic;
    }
    else if (value == std::string("NicelySystematic"))
    {
        method.method = ResampleType::NicelySystematic;
    }
    else if (value == std::string("SortSystematic"))
    {
        method.method = ResampleType::SortSystematic;
    }
    else if (value == std::string("Stratified"))
    {
        method.method = ResampleType::Stratified;
    }
}