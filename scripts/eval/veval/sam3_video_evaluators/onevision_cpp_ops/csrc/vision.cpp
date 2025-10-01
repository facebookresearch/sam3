// Copyright (c) Facebook, Inc. and its affiliates.
// Imported from
// https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/csrc/vision.cpp

#include <sstream>
#include <stdexcept>
#include <type_traits>
#include "cocoeval/cocoeval.h"

namespace onevision_cpp {

std::string boolToBinary(const std::vector<bool>& vec) {
  if (vec.size() == 0) {
    return "null";
  }
  std::string result;
  for (const auto& b : vec) {
    result += b ? '1' : '0';
  }
  return result;
}

std::vector<bool> binaryToBool(const std::string& str) {
  if (str == "null") {
    return std::vector<bool>();
  }
  std::vector<bool> result;
  result.reserve(str.size());
  for (char c : str) {
    result.push_back(c == '1');
  }
  return result;
}

template <typename T>
std::string serialize_sparse(const std::vector<T>& data) {
  static_assert(
      std::is_fundamental<T>::value, "Only primitive types are supported");

  std::stringstream ss;
  ss << data.size() << " "; // size of the array
  size_t non_zero_count = 0;

  // count non-zero entries
  for (const auto& d : data) {
    if (d != T(0)) {
      non_zero_count++;
    }
  }

  ss << non_zero_count << " "; // number of non-zero entries

  // serialize non-zero entries
  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i] != T(0)) {
      ss << i << " " << data[i] << " "; // index and value
    }
  }

  return ss.str();
}
template <typename T>
void deserialize_sparse(std::istringstream& ss, std::vector<T>* data) {
  static_assert(
      std::is_fundamental<T>::value, "Only primitive types are supported");

  size_t size;
  ss >> size; // read size of the array

  size_t non_zero_count;
  ss >> non_zero_count; // read number of non-zero entries

  // resize data vector
  data->resize(size, T(0));

  // read non-zero entries
  for (size_t i = 0; i < non_zero_count; ++i) {
    size_t index;
    T value;
    ss >> index >> value; // read index and value
    if (index >= size) {
      throw std::runtime_error("Index out of bounds");
    }
    (*data)[index] = value;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("COCOevalAccumulate", &COCOeval::Accumulate, "COCOeval::Accumulate");
  m.def(
      "COCOevalEvaluateImages",
      &COCOeval::EvaluateImages,
      "COCOeval::EvaluateImages");
  pybind11::class_<COCOeval::InstanceAnnotation>(m, "InstanceAnnotation")
      .def(pybind11::init<uint64_t, double, double, bool, bool>());
  pybind11::class_<COCOeval::ImageEvaluation>(m, "ImageEvaluation")
      .def(pybind11::init<>())
      .def(py::pickle(
          [](const COCOeval::ImageEvaluation& p) { // __getstate__
            /* Return a byte that fully encodes the state of the object */
            std::ostringstream oss;
            oss << serialize_sparse(p.detection_matches);

            oss << p.detection_scores.size() << " ";
            for (auto s : p.detection_scores) {
              oss << s << " ";
            }

            oss << boolToBinary(p.ground_truth_ignores) << " ";
            oss << boolToBinary(p.detection_ignores) << " ";
            return py::bytes(oss.str());
          },
          [](py::bytes t) { // __setstate__
            /* Create a new C++ instance */
            COCOeval::ImageEvaluation p;
            std::istringstream iss(t);

            deserialize_sparse(iss, &p.detection_matches);

            uint64_t num_detection_scores;
            iss >> num_detection_scores;
            p.detection_scores.resize(num_detection_scores);
            for (auto& s : p.detection_scores) {
              iss >> s;
            }

            std::string ground_truth_ignores_str;
            iss >> ground_truth_ignores_str;
            p.ground_truth_ignores = binaryToBool(ground_truth_ignores_str);

            std::string detection_ignores_str;
            iss >> detection_ignores_str;
            p.detection_ignores = binaryToBool(detection_ignores_str);
            return p;
          }));
}
} // namespace onevision_cpp
