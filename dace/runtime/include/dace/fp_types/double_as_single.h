#pragma once

#include <cmath>

#include "types.h"

namespace dace {

/**
 * A double precision type that uses Dekkers algorithm for addition
 * to maintain higher precision intermediate results altough using single precision
 */
class double_as_single {
 private:
  float higher;
  float lower;

  DACE_HDFI void decompose(double val) {
    higher = static_cast<float>(val);
    lower = static_cast<float>(val - higher);
  }

 public:
  DACE_HDFI double_as_single() : higher(0.0f), lower(0.0f) {}
  
  DACE_HDFI double_as_single(double v) {
    decompose(v);
  }
  
  DACE_HDFI double_as_single(float v) : higher(v), lower(0.0f) {}
  
  DACE_HDFI double_as_single(int v) {
    decompose(static_cast<double>(v));
  }

  DACE_HDFI double_as_single(const double_as_single& other)
      : higher(other.higher), lower(other.lower) {}
      
  DACE_HDFI double_as_single& operator=(const double_as_single& other) {
    higher = other.higher;
    lower = other.lower;
    return *this;
  }
  
  DACE_HDFI double_as_single& operator=(double v) {
    decompose(v);
    return *this;
  }

  DACE_HDFI operator double() const { 
    return static_cast<double>(higher) + static_cast<double>(lower); 
  }
  
  DACE_HDFI operator float() const { 
    return higher + lower; 
  }

  DACE_HDFI double_as_single operator+(const double_as_single& other) const {
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
    return double_as_single(z);
  }

  DACE_HDFI double_as_single operator-(const double_as_single& other) const {
    double this_val = static_cast<double>(higher) + static_cast<double>(lower);
    double other_val = static_cast<double>(other.higher) + static_cast<double>(other.lower);
    return double_as_single(this_val - other_val);
  }

  DACE_HDFI double_as_single operator*(const double_as_single& other) const {
    double this_val = static_cast<double>(higher) + static_cast<double>(lower);
    double other_val = static_cast<double>(other.higher) + static_cast<double>(other.lower);
    return double_as_single(this_val * other_val);
  }

  DACE_HDFI double_as_single operator/(const double_as_single& other) const {
    double this_val = static_cast<double>(higher) + static_cast<double>(lower);
    double other_val = static_cast<double>(other.higher) + static_cast<double>(other.lower);
    return double_as_single(this_val / other_val);
  }

  DACE_HDFI double_as_single& operator+=(const double_as_single& other) {
    *this = *this + other;
    return *this;
  }

  DACE_HDFI double_as_single& operator-=(const double_as_single& other) {
    *this = *this - other;
    return *this;
  }

  DACE_HDFI double_as_single& operator*=(const double_as_single& other) {
    *this = *this * other;
    return *this;
  }

  DACE_HDFI double_as_single& operator/=(const double_as_single& other) {
    *this = *this / other;
    return *this;
  }

  DACE_HDFI bool operator==(const double_as_single& other) const {
    return higher == other.higher && lower == other.lower;
  }

  DACE_HDFI bool operator!=(const double_as_single& other) const {
    return !(*this == other);
  }

  DACE_HDFI bool operator<(const double_as_single& other) const {
    double this_val = static_cast<double>(higher) + static_cast<double>(lower);
    double other_val = static_cast<double>(other.higher) + static_cast<double>(other.lower);
    return this_val < other_val;
  }

  DACE_HDFI bool operator<=(const double_as_single& other) const {
    return *this < other || *this == other;
  }

  DACE_HDFI bool operator>(const double_as_single& other) const {
    return !(*this <= other);
  }

  DACE_HDFI bool operator>=(const double_as_single& other) const {
    return !(*this < other);
  }

  DACE_HDFI double_as_single operator-() const {
    return double_as_single(-(static_cast<double>(higher) + static_cast<double>(lower)));
  }

  DACE_HDFI double_as_single operator+() const { return *this; }

  DACE_HDFI double get_value() const { 
    return static_cast<double>(higher) + static_cast<double>(lower); 
  }
};

DACE_HDFI double_as_single operator+(double lhs, const double_as_single& rhs) {
  return double_as_single(lhs) + rhs;
}

DACE_HDFI double_as_single operator+(const double_as_single& lhs, double rhs) {
  return lhs + double_as_single(rhs);
}

}