#pragma once

#include <cmath>

#include "../definitions.h"

namespace dace {

/**
 * A double precision type that uses Dekkers algorithm for addition
 * to maintain higher precision intermediate results altough using single precision
 */
class simulated_double {
 private:
  float higher;
  float lower;

  DACE_HDFI void decompose(double val) {
    higher = static_cast<float>(val);
    lower = static_cast<float>(val - higher);
  }

 public:
  DACE_HDFI simulated_double() : higher(0.0f), lower(0.0f) {}

  DACE_HDFI simulated_double(double v) {
    decompose(v);
  }

  DACE_HDFI simulated_double(float v) : higher(v), lower(0.0f) {}

  DACE_HDFI simulated_double(int v) {
    decompose(static_cast<double>(v));
  }

  DACE_HDFI simulated_double(const simulated_double& other)
      : higher(other.higher), lower(other.lower) {}

  DACE_HDFI simulated_double& operator=(const simulated_double& other) {
    higher = other.higher;
    lower = other.lower;
    return *this;
  }

  DACE_HDFI simulated_double& operator=(double v) {
    decompose(v);
    return *this;
  }

  DACE_HDFI operator double() const {
    return static_cast<double>(higher) + static_cast<double>(lower);
  }

  DACE_HDFI operator float() const {
    return higher + lower;
  }

  DACE_HDFI simulated_double operator+(const simulated_double& other) const {
    float x = this->higher;
    float xx = this->lower;
    float y = other.higher;
    float yy = other.lower;

    float r, s;
    r = x + y;
    if (fabsf(x) > fabsf(y)) {
      s = x - r + y + yy + xx;
    } else {
      s = y - r + x + xx + yy;
    }
    double z = static_cast<double>(r) + static_cast<double>(s);
    return simulated_double(z);
  }

  DACE_HDFI simulated_double operator-(const simulated_double& other) const {
    double this_val = static_cast<double>(higher) + static_cast<double>(lower);
    double other_val = static_cast<double>(other.higher) + static_cast<double>(other.lower);
    return simulated_double(this_val - other_val);
  }

  DACE_HDFI simulated_double operator*(const simulated_double& other) const {
    double this_val = static_cast<double>(higher) + static_cast<double>(lower);
    double other_val = static_cast<double>(other.higher) + static_cast<double>(other.lower);
    return simulated_double(this_val * other_val);
  }

  DACE_HDFI simulated_double operator/(const simulated_double& other) const {
    double this_val = static_cast<double>(higher) + static_cast<double>(lower);
    double other_val = static_cast<double>(other.higher) + static_cast<double>(other.lower);
    return simulated_double(this_val / other_val);
  }

  DACE_HDFI simulated_double& operator+=(const simulated_double& other) {
    *this = *this + other;
    return *this;
  }

  DACE_HDFI simulated_double& operator-=(const simulated_double& other) {
    *this = *this - other;
    return *this;
  }

  DACE_HDFI simulated_double& operator*=(const simulated_double& other) {
    *this = *this * other;
    return *this;
  }

  DACE_HDFI simulated_double& operator/=(const simulated_double& other) {
    *this = *this / other;
    return *this;
  }

  DACE_HDFI bool operator==(const simulated_double& other) const {
    return higher == other.higher && lower == other.lower;
  }

  DACE_HDFI bool operator!=(const simulated_double& other) const {
    return !(*this == other);
  }

  DACE_HDFI bool operator<(const simulated_double& other) const {
    double this_val = static_cast<double>(higher) + static_cast<double>(lower);
    double other_val = static_cast<double>(other.higher) + static_cast<double>(other.lower);
    return this_val < other_val;
  }

  DACE_HDFI bool operator<=(const simulated_double& other) const {
    return *this < other || *this == other;
  }

  DACE_HDFI bool operator>(const simulated_double& other) const {
    return !(*this <= other);
  }

  DACE_HDFI bool operator>=(const simulated_double& other) const {
    return !(*this < other);
  }

  DACE_HDFI simulated_double operator-() const {
    return simulated_double(-(static_cast<double>(higher) + static_cast<double>(lower)));
  }

  DACE_HDFI simulated_double operator+() const { return *this; }

  DACE_HDFI double get_value() const {
    return static_cast<double>(higher) + static_cast<double>(lower);
  }
};

DACE_HDFI simulated_double operator+(double lhs, const simulated_double& rhs) {
  return simulated_double(lhs) + rhs;
}

DACE_HDFI simulated_double operator+(const simulated_double& lhs, double rhs) {
  return lhs + simulated_double(rhs);
}

}