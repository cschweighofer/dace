#pragma once

#include <cmath>
#include <algorithm>
#include <gmp.h>

#include "../definitions.h"

namespace dace {

/**
 * CURRENTLY NOT WORKING
 * A wrapper class for GNU Multiple Precision Rational (MPQ) type.
 * This provides arbitrary-precision rational arithmetic using the GMP library.
 * 
 * Note: This requires the GMP library to be linked. Compile with -lgmp.
 */
class mpq_class {
private:
  mpq_t value;

  // Canonicalize the fraction (simplified - no GCD for now)
  DACE_HDFI void canonicalize() {
    mpq_canonicalize(value);
  }

public:
  // Default constructor: 0/1
  DACE_HDFI mpq_class() {
    mpq_init(value);
    mpq_set_ui(value, 0, 1);
  }

  // Constructor from numerator and denominator
  DACE_HDFI mpq_class(long long num, long long den) {
    mpq_init(value);
    if (den == 0) {
      // Handle division by zero - set to NaN representation
      mpq_set_ui(value, 0, 1);
      return;
    }
    mpq_set_si(value, num, 1);
    mpz_set_si(mpq_denref(value), den);
    canonicalize();
  }

  // Constructor from long long
  DACE_HDFI mpq_class(long long value) {
    mpq_init(this->value);
    mpq_set_si(this->value, value, 1);
  }

  // Constructor from integer
  DACE_HDFI mpq_class(int value) {
    mpq_init(this->value);
    mpq_set_si(this->value, value, 1);
  }

  // Constructor from double (higher precision conversion)
  DACE_HDFI mpq_class(double value) {
    mpq_init(this->value);
    if (std::isnan(value) || std::isinf(value)) {
      mpq_set_ui(this->value, 0, 1);
      return;
    }
    
    // Handle zero case
    if (value == 0.0) {
      mpq_set_ui(this->value, 0, 1);
      return;
    }
    
    // Use conservative scale to balance precision vs overflow risk
    const long long scale = 10000LL;  // 10^4 - much more conservative
    long long numerator = static_cast<long long>(value * scale);
    mpq_set_si(this->value, numerator, scale);
    canonicalize();
  }

  // Copy constructor
  DACE_HDFI mpq_class(const mpq_class& other) {
    mpq_init(value);
    mpq_set(value, other.value);
  }

  // Destructor
  DACE_HDFI ~mpq_class() {
    // mpq_clear(value);
  }

  // Assignment operator
  DACE_HDFI mpq_class& operator=(const mpq_class& other) {
    if (this != &other) {
      mpq_set(value, other.value);
    }
    return *this;
  }

  // Assignment from double
  DACE_HDFI mpq_class& operator=(double value) {
    *this = mpq_class(value);
    return *this;
  }

  // Assignment from long long
  DACE_HDFI mpq_class& operator=(long long val) {
    mpq_set_si(value, val, 1);
    return *this;
  }

  // Conversion to double
  DACE_HDFI operator double() const {
    return mpq_get_d(value);
  }

  // Conversion to float
  DACE_HDFI operator float() const {
    return static_cast<float>(mpq_get_d(value));
  }

  // Arithmetic operators
  DACE_HDFI mpq_class operator+(const mpq_class& other) const {
    mpq_class result;
    mpq_add(result.value, value, other.value);
    return result;
  }

  DACE_HDFI mpq_class operator-(const mpq_class& other) const {
    mpq_class result;
    mpq_sub(result.value, value, other.value);
    return result;
  }

  DACE_HDFI mpq_class operator*(const mpq_class& other) const {
    mpq_class result;
    mpq_mul(result.value, value, other.value);
    return result;
  }

  DACE_HDFI mpq_class operator/(const mpq_class& other) const {
    mpq_class result;
    if (mpq_sgn(other.value) == 0) {
      // Division by zero - return NaN representation
      mpq_set_ui(result.value, 0, 1);
    } else {
      mpq_div(result.value, value, other.value);
    }
    return result;
  }

  // Compound assignment operators
  DACE_HDFI mpq_class& operator+=(const mpq_class& other) {
    mpq_add(value, value, other.value);
    return *this;
  }

  DACE_HDFI mpq_class& operator-=(const mpq_class& other) {
    mpq_sub(value, value, other.value);
    return *this;
  }

  DACE_HDFI mpq_class& operator*=(const mpq_class& other) {
    mpq_mul(value, value, other.value);
    return *this;
  }

  DACE_HDFI mpq_class& operator/=(const mpq_class& other) {
    if (mpq_sgn(other.value) == 0) {
      // Division by zero - set to NaN representation
      mpq_set_ui(value, 0, 1);
    } else {
      mpq_div(value, value, other.value);
    }
    return *this;
  }

  // Comparison operators
  DACE_HDFI bool operator==(const mpq_class& other) const {
    return mpq_equal(value, other.value) != 0;
  }

  DACE_HDFI bool operator!=(const mpq_class& other) const {
    return !(*this == other);
  }

  DACE_HDFI bool operator<(const mpq_class& other) const {
    return mpq_cmp(value, other.value) < 0;
  }

  DACE_HDFI bool operator<=(const mpq_class& other) const {
    return mpq_cmp(value, other.value) <= 0;
  }

  DACE_HDFI bool operator>(const mpq_class& other) const {
    return mpq_cmp(value, other.value) > 0;
  }

  DACE_HDFI bool operator>=(const mpq_class& other) const {
    return mpq_cmp(value, other.value) >= 0;
  }

  // Unary operators
  DACE_HDFI mpq_class operator-() const {
    mpq_class result;
    mpq_neg(result.value, value);
    return result;
  }

  DACE_HDFI mpq_class operator+() const {
    return *this;
  }

  // Utility functions
  DACE_HDFI bool is_zero() const {
    return mpq_sgn(value) == 0;
  }

  DACE_HDFI bool is_negative() const {
    return mpq_sgn(value) < 0;
  }

  DACE_HDFI bool is_positive() const {
    return mpq_sgn(value) > 0;
  }

  // Get numerator and denominator as strings for debugging
  DACE_HDFI const char* get_numerator_str() const {
    return mpz_get_str(nullptr, 10, mpq_numref(value));
  }

  DACE_HDFI const char* get_denominator_str() const {
    return mpz_get_str(nullptr, 10, mpq_denref(value));
  }

  // Friend function for abs
  friend DACE_HDFI mpq_class abs(const mpq_class& x);
};

// Free functions for arithmetic with scalars
DACE_HDFI mpq_class operator+(long long lhs, const mpq_class& rhs) {
  return mpq_class(lhs) + rhs;
}

DACE_HDFI mpq_class operator+(const mpq_class& lhs, long long rhs) {
  return lhs + mpq_class(rhs);
}

DACE_HDFI mpq_class operator-(long long lhs, const mpq_class& rhs) {
  return mpq_class(lhs) - rhs;
}

DACE_HDFI mpq_class operator-(const mpq_class& lhs, long long rhs) {
  return lhs - mpq_class(rhs);
}

DACE_HDFI mpq_class operator*(long long lhs, const mpq_class& rhs) {
  return mpq_class(lhs) * rhs;
}

DACE_HDFI mpq_class operator*(const mpq_class& lhs, long long rhs) {
  return lhs * mpq_class(rhs);
}

DACE_HDFI mpq_class operator/(long long lhs, const mpq_class& rhs) {
  return mpq_class(lhs) / rhs;
}

DACE_HDFI mpq_class operator/(const mpq_class& lhs, long long rhs) {
  return lhs / mpq_class(rhs);
}

// Math functions
DACE_HDFI mpq_class abs(const mpq_class& x) {
  mpq_class result;
  mpq_abs(result.value, x.value);
  return result;
}

DACE_HDFI mpq_class min(const mpq_class& a, const mpq_class& b) {
  return (a < b) ? a : b;
}

DACE_HDFI mpq_class max(const mpq_class& a, const mpq_class& b) {
  return (a > b) ? a : b;
}

} // namespace dace
