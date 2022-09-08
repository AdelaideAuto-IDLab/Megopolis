#pragma once

#include "../util/json_serialize.hpp"
#include <array>
#include <vector>
#include <utility>

namespace logging
{
    struct TargetGroundTruth {
        size_t id;
        std::array<float, 2> pos;

        TargetGroundTruth(size_t id, std::array<float, 2> pos) : id(id), pos(pos) {}

        template<typename WRITER>
        void serialize(WRITER &writer) {
            json_se::open_brace(writer);
            json_se::serialize_field(writer, "id", id);
            json_se::comma(writer);
            json_se::serialize_field_name(writer, "pos");
            json_se::serialize_array(writer, pos.data(), 2);
            json_se::close_brace(writer);
        }
    };

    struct UavPos {
        std::array<float, 3> pos;
        float yaw;

        UavPos() {}
        UavPos(std::array<float, 3> pos, float yaw) : pos(pos), yaw(yaw) {}

        template<typename WRITER>
        void serialize(WRITER &writer) {
            json_se::open_brace(writer);
            json_se::serialize_field_name(writer, "pos");
            json_se::serialize_array(writer, pos.data(), 3);
            json_se::comma(writer);
            json_se::serialize_field(writer, "yaw", yaw);
            json_se::close_brace(writer);
        }
    };

    struct Target {
        size_t id;
        std::vector<std::array<float, 2>> particles;
        std::array<float, 2> estimate;
        float stddev;

        Target(
            size_t id, 
            std::vector<std::array<float, 2>> particles, 
            std::array<float, 2> estimate, 
            float stddev
        ) :
            id(id),
            particles(particles),
            estimate(estimate),
            stddev(stddev)
        {}

        template<typename WRITER>
        void serialize(WRITER &writer) {
            json_se::open_brace(writer);
            json_se::serialize_field(writer, "id", id);
            json_se::comma(writer);
            json_se::serialize_field_name(writer, "particles");
            json_se::open_bracket(writer);

            for (size_t i = 1; i < particles.size(); i++) {
                json_se::serialize_array(writer, particles[i - 1].data(), 2);
                json_se::comma(writer);
            }

            if (particles.size() > 0) {
                json_se::serialize_array(writer, particles.back().data(), 2);
            }

            json_se::close_bracket(writer);
            json_se::comma(writer);
            json_se::serialize_field_name(writer, "estimate");
            json_se::serialize_array(writer, estimate.data(), 2);
            json_se::comma(writer);
            json_se::serialize_field(writer, "stddev", stddev);
            json_se::close_brace(writer);
        }
    };
    
    enum class MessageType {
        UavPos,
        Target,
        TargetGroundTruth
    };

    struct LogMessage {
        MessageType type;
        union {
            UavPos uav;
            Target target;
            TargetGroundTruth ground_truth;
        };

        LogMessage() : type(MessageType::UavPos), uav(UavPos()) {}
        LogMessage(UavPos data) : type(MessageType::UavPos), uav(data) {}
        LogMessage(Target&& data) :  type(MessageType::Target), target(data) {}
        LogMessage(TargetGroundTruth data) : type(MessageType::TargetGroundTruth), ground_truth(data) {}
        ~LogMessage() {
            if (type == MessageType::Target) {
                target.~Target(); 
            }
        }

        LogMessage(const LogMessage& other) : type(other.type) {
            if (other.type == MessageType::UavPos) {
                uav = other.uav;
            }
            else if (other.type == MessageType::TargetGroundTruth) {
                ground_truth = other.ground_truth;
            }
            else if (other.type == MessageType::Target) {
                new (&target) auto(other.target);
            }
        } 

        LogMessage(LogMessage&& other) : type(other.type) {
            if (other.type == MessageType::UavPos) {
                uav = other.uav;
            }
            else if (other.type == MessageType::TargetGroundTruth) {
                ground_truth = other.ground_truth;
            }
            else if (other.type == MessageType::Target) {
                new (&target) auto(std::move(other.target));
            }
        }

        LogMessage& operator=(const LogMessage& other) {
            if (type == MessageType::Target) {
                if (other.type == MessageType::Target) {
                    target = other.target;
                    return *this;
                }
                else {
                    target.~Target(); 
                }
            }

            type = other.type;
            if (other.type == MessageType::UavPos) {
                uav = other.uav;
            }
            else if (other.type == MessageType::TargetGroundTruth) {
                ground_truth = other.ground_truth;
            }
            else {
                target = other.target;
            }

            return *this;
        }

        LogMessage& operator=(LogMessage&& other) {
            if (type == MessageType::Target) {
                if (other.type == MessageType::Target) {
                    target = std::move(other.target);
                    return *this;
                }
                else {
                    target.~Target(); 
                }
            }

            type = other.type;
            if (other.type == MessageType::UavPos) {
                uav = other.uav;
            }
            else if (other.type == MessageType::TargetGroundTruth) {
                ground_truth = other.ground_truth;
            }
            else {
                target = std::move(other.target);
            }

            return *this;
        }

        template<typename WRITER>
        void serialize(WRITER &writer) {
            json_se::open_brace(writer);

            if (type == MessageType::UavPos) {
                json_se::serialize_field_name(writer, "UavPos");
                uav.serialize(writer);
            }
            else if (type == MessageType::TargetGroundTruth) {
                json_se::serialize_field_name(writer, "AnimalGroundTruth");
                ground_truth.serialize(writer);
            }
            else {
                json_se::serialize_field_name(writer, "TrackedTarget");
                target.serialize(writer);
            }

            json_se::close_brace(writer);
        }
    };

};

