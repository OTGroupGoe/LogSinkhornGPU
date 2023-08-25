#include <iostream>
#include <sstream>
#include <string>
#include <torch/extension.h>

class Formatter {
public:
  Formatter() {}
  ~Formatter() {}

  template <typename Type> Formatter &operator<<(const Type &value) {
    stream_ << value;
    return *this;
  }

  std::string str() const { return stream_.str(); }
  operator std::string() const { return stream_.str(); }

  enum ConvertToString { to_str };

  std::string operator>>(ConvertToString) { return stream_.str(); }

private:
  std::stringstream stream_;
  Formatter(const Formatter &);
  Formatter &operator=(Formatter &);
};

// Template struct to determine the appropriate tensor type based on the input 
template <typename Dtype>
struct TensorTypeSelector {
    static const torch::ScalarType type = torch::kFloat32;
};

// Specialize template for doubles
template <>
struct TensorTypeSelector<double> {
    static const torch::ScalarType type = torch::kFloat64;
};