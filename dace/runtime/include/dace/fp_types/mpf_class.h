#pragma once

#include <cmath>
#include <algorithm>
#include <gmp.h>

#include "../definitions.h"

namespace dace {

/**
 * A wrapper class for GNU Multiple Precision Floating-point (MPF) type.
 * This provides arbitrary-precision floating-point arithmetic using the GMP library.
 * 
 * Note: This requires the GMP library to be linked. Compile with -lgmp.
 */
class mpf_class {
private:
  mpf_t value;
  bool initialized;
  
  // Default precision in bits (can be adjusted)
  static constexpr mp_bitcnt_t DEFAULT_PRECISION = 256;

public:
  // Default constructor: 0.0
  DACE_HDFI mpf_class() : initialized(false) {
    mpf_init2(value, DEFAULT_PRECISION);
    mpf_set_ui(value, 0);
    initialized = true;
  }

  // Constructor with custom precision
  DACE_HDFI mpf_class(mp_bitcnt_t precision) : initialized(false) {
    mpf_init2(value, precision);
    mpf_set_ui(value, 0);
    initialized = true;
  }

  // Constructor from long long
  DACE_HDFI mpf_class(long long value) : initialized(false) {
    mpf_init2(this->value, DEFAULT_PRECISION);
    mpf_set_si(this->value, value);
    initialized = true;
  }

  // Constructor from integer
  DACE_HDFI mpf_class(int value) : initialized(false) {
    mpf_init2(this->value, DEFAULT_PRECISION);
    mpf_set_si(this->value, value);
    initialized = true;
  }

  // Constructor from double
  DACE_HDFI mpf_class(double value) : initialized(false) {
    mpf_init2(this->value, DEFAULT_PRECISION);
    if (std::isnan(value) || std::isinf(value)) {
      mpf_set_ui(this->value, 0);
    } else {
      mpf_set_d(this->value, value);
    }
    initialized = true;
  }

  // Constructor from string (for high precision initialization)
  DACE_HDFI mpf_class(const char* str) : initialized(false) {
    mpf_init2(value, DEFAULT_PRECISION);
    mpf_set_str(value, str, 10);
    initialized = true;
  }

  // Copy constructor
  DACE_HDFI mpf_class(const mpf_class& other) : initialized(false) {
    mpf_init2(value, mpf_get_prec(other.value));
    mpf_set(value, other.value);
    initialized = true;
  }

  // Destructor
  DACE_HDFI ~mpf_class() {
    if (initialized) {
      mpf_clear(value);
      initialized = false;
    }
  }

  // Assignment operator
  DACE_HDFI mpf_class& operator=(const mpf_class& other) {
    if (this != &other) {
      if (!initialized) {
        mpf_init2(value, mpf_get_prec(other.value));
        initialized = true;
      }
      mpf_set(value, other.value);
    }
    return *this;
  }

  // Assignment from double
  DACE_HDFI mpf_class& operator=(double val) {
    if (!initialized) {
      mpf_init2(value, DEFAULT_PRECISION);
      initialized = true;
    }
    if (std::isnan(val) || std::isinf(val)) {
      mpf_set_ui(value, 0);
    } else {
      mpf_set_d(value, val);
    }
    return *this;
  }

  // Assignment from long long
  DACE_HDFI mpf_class& operator=(long long val) {
    if (!initialized) {
      mpf_init2(value, DEFAULT_PRECISION);
      initialized = true;
    }
    mpf_set_si(value, val);
    return *this;
  }

  // Conversion to double
  DACE_HDFI operator double() const {
    return mpf_get_d(value);
  }

  // Conversion to float
  DACE_HDFI operator float() const {
    return static_cast<float>(mpf_get_d(value));
  }

  // Arithmetic operators
  DACE_HDFI mpf_class operator+(const mpf_class& other) const {
    mpf_class result;
    mpf_add(result.value, value, other.value);
    return result;
  }

  DACE_HDFI mpf_class operator-(const mpf_class& other) const {
    mpf_class result;
    mpf_sub(result.value, value, other.value);
    return result;
  }

  DACE_HDFI mpf_class operator*(const mpf_class& other) const {
    mpf_class result;
    mpf_mul(result.value, value, other.value);
    return result;
  }

  DACE_HDFI mpf_class operator/(const mpf_class& other) const {
    mpf_class result;
    if (mpf_sgn(other.value) == 0) {
      // Division by zero - return NaN representation
      mpf_set_ui(result.value, 0);
    } else {
      mpf_div(result.value, value, other.value);
    }
    return result;
  }

  // Compound assignment operators
  DACE_HDFI mpf_class& operator+=(const mpf_class& other) {
    mpf_add(value, value, other.value);
    return *this;
  }

  DACE_HDFI mpf_class& operator-=(const mpf_class& other) {
    mpf_sub(value, value, other.value);
    return *this;
  }

  DACE_HDFI mpf_class& operator*=(const mpf_class& other) {
    mpf_mul(value, value, other.value);
    return *this;
  }

  DACE_HDFI mpf_class& operator/=(const mpf_class& other) {
    if (mpf_sgn(other.value) == 0) {
      // Division by zero - set to NaN representation
      mpf_set_ui(value, 0);
    } else {
      mpf_div(value, value, other.value);
    }
    return *this;
  }

  // Comparison operators
  DACE_HDFI bool operator==(const mpf_class& other) const {
    return mpf_cmp(value, other.value) == 0;
  }

  DACE_HDFI bool operator!=(const mpf_class& other) const {
    return !(*this == other);
  }

  DACE_HDFI bool operator<(const mpf_class& other) const {
    return mpf_cmp(value, other.value) < 0;
  }

  DACE_HDFI bool operator<=(const mpf_class& other) const {
    return mpf_cmp(value, other.value) <= 0;
  }

  DACE_HDFI bool operator>(const mpf_class& other) const {
    return mpf_cmp(value, other.value) > 0;
  }

  DACE_HDFI bool operator>=(const mpf_class& other) const {
    return mpf_cmp(value, other.value) >= 0;
  }

  // Unary operators
  DACE_HDFI mpf_class operator-() const {
    mpf_class result;
    mpf_neg(result.value, value);
    return result;
  }

  DACE_HDFI mpf_class operator+() const {
    return *this;
  }

  // Utility functions
  DACE_HDFI bool is_zero() const {
    return mpf_sgn(value) == 0;
  }

  DACE_HDFI bool is_negative() const {
    return mpf_sgn(value) < 0;
  }

  DACE_HDFI bool is_positive() const {
    return mpf_sgn(value) > 0;
  }

  // Get precision in bits
  DACE_HDFI mp_bitcnt_t get_precision() const {
    return mpf_get_prec(value);
  }

  // Set precision in bits
  DACE_HDFI void set_precision(mp_bitcnt_t precision) {
    mpf_set_prec(value, precision);
  }

  // Get string representation for debugging
  DACE_HDFI const char* get_str() const {
    return mpf_get_str(nullptr, nullptr, 10, 0, value);
  }

  // Friend functions for mathematical operations
  friend DACE_HDFI mpf_class abs(const mpf_class& x);
  friend DACE_HDFI mpf_class sqrt(const mpf_class& x);
  friend DACE_HDFI mpf_class mpf_pow(const mpf_class& base, unsigned long exp);
};

// Free functions for arithmetic with scalars
DACE_HDFI mpf_class operator+(long long lhs, const mpf_class& rhs) {
  return mpf_class(lhs) + rhs;
}

DACE_HDFI mpf_class operator+(const mpf_class& lhs, long long rhs) {
  return lhs + mpf_class(rhs);
}

DACE_HDFI mpf_class operator+(double lhs, const mpf_class& rhs) {
  return mpf_class(lhs) + rhs;
}

DACE_HDFI mpf_class operator+(const mpf_class& lhs, double rhs) {
  return lhs + mpf_class(rhs);
}

DACE_HDFI mpf_class operator-(long long lhs, const mpf_class& rhs) {
  return mpf_class(lhs) - rhs;
}

DACE_HDFI mpf_class operator-(const mpf_class& lhs, long long rhs) {
  return lhs - mpf_class(rhs);
}

DACE_HDFI mpf_class operator-(double lhs, const mpf_class& rhs) {
  return mpf_class(lhs) - rhs;
}

DACE_HDFI mpf_class operator-(const mpf_class& lhs, double rhs) {
  return lhs - mpf_class(rhs);
}

DACE_HDFI mpf_class operator*(long long lhs, const mpf_class& rhs) {
  return mpf_class(lhs) * rhs;
}

DACE_HDFI mpf_class operator*(const mpf_class& lhs, long long rhs) {
  return lhs * mpf_class(rhs);
}

DACE_HDFI mpf_class operator*(double lhs, const mpf_class& rhs) {
  return mpf_class(lhs) * rhs;
}

DACE_HDFI mpf_class operator*(const mpf_class& lhs, double rhs) {
  return lhs * mpf_class(rhs);
}

DACE_HDFI mpf_class operator/(long long lhs, const mpf_class& rhs) {
  return mpf_class(lhs) / rhs;
}

DACE_HDFI mpf_class operator/(const mpf_class& lhs, long long rhs) {
  return lhs / mpf_class(rhs);
}

DACE_HDFI mpf_class operator/(double lhs, const mpf_class& rhs) {
  return mpf_class(lhs) / rhs;
}

DACE_HDFI mpf_class operator/(const mpf_class& lhs, double rhs) {
  return lhs / mpf_class(rhs);
}

// Math functions
DACE_HDFI mpf_class abs(const mpf_class& x) {
  mpf_class result;
  mpf_abs(result.value, x.value);
  return result;
}

DACE_HDFI mpf_class min(const mpf_class& a, const mpf_class& b) {
  return (a < b) ? a : b;
}

DACE_HDFI mpf_class max(const mpf_class& a, const mpf_class& b) {
  return (a > b) ? a : b;
}

// Additional mathematical functions
DACE_HDFI mpf_class sqrt(const mpf_class& x) {
  mpf_class result;
  if (mpf_sgn(x.value) < 0) {
    // Negative number - return NaN representation
    mpf_set_ui(result.value, 0);
  } else {
    mpf_sqrt(result.value, x.value);
  }
  return result;
}

DACE_HDFI mpf_class mpf_pow(const mpf_class& base, unsigned long exp) {
  mpf_class result;
  mpf_pow_ui(result.value, base.value, exp);
  return result;
}

} // namespace dace
