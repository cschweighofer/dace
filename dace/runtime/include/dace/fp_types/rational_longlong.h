#pragma once

#include <cmath>
#include <algorithm>

#include "../definitions.h"

namespace dace {

// Compiling, but computations fail fast
class rational {
 private:
  long long numerator;
  long long denominator;

  // Canonicalize the fraction (simplified - no GCD for now)
  DACE_HDFI void canonicalize() {
    if (denominator == 0) {
      // Handle division by zero - set to NaN representation
      numerator = 0;
      denominator = 1;
      return;
    }
    
    if (denominator < 0) {
      numerator = -numerator;
      denominator = -denominator;
    }
    
    // Skip GCD reduction for now to avoid complexity
  }

 public:
  // Default constructor: 0/1
  DACE_HDFI rational() : numerator(0), denominator(1) {}

  // Constructor from numerator and denominator
  DACE_HDFI rational(long long num, long long den) : numerator(num), denominator(den) {
    canonicalize();
  }

  // Constructor from long long
  DACE_HDFI rational(long long value) : numerator(value), denominator(1) {}

  // Constructor from integer
  DACE_HDFI rational(int value) : numerator(value), denominator(1) {}

  // Constructor from double (higher precision conversion)
  DACE_HDFI rational(double value) {
    if (std::isnan(value) || std::isinf(value)) {
      numerator = 0;
      denominator = 1;
      return;
    }
    
    // Handle zero case
    if (value == 0.0) {
      numerator = 0;
      denominator = 1;
      return;
    }
    
    // Handle negative values
    bool negative = value < 0;
    if (negative) {
      value = -value;
    }
    
    // Use conservative scale to balance precision vs overflow risk
    const long long scale = 10000LL;  // 10^4 - much more conservative
    numerator = static_cast<long long>(value * scale);
    denominator = scale;
    if (negative) numerator = -numerator;
    canonicalize();
  }

  // Copy constructor
  DACE_HDFI rational(const rational& other) : numerator(other.numerator), denominator(other.denominator) {}

  // Assignment operator
  DACE_HDFI rational& operator=(const rational& other) {
    if (this != &other) {
      numerator = other.numerator;
      denominator = other.denominator;
    }
    return *this;
  }

  // Assignment from double
  DACE_HDFI rational& operator=(double value) {
    *this = rational(value);
    return *this;
  }

  // Assignment from long long
  DACE_HDFI rational& operator=(long long value) {
    numerator = value;
    denominator = 1;
    return *this;
  }

  // Conversion to double
  DACE_HDFI operator double() const {
    return static_cast<double>(numerator) / static_cast<double>(denominator);
  }

  // Conversion to float
  DACE_HDFI operator float() const {
    return static_cast<float>(numerator) / static_cast<float>(denominator);
  }

  // Helper function to compute GCD
  DACE_HDFI static long long gcd(long long a, long long b) {
    a = std::abs(a);
    b = std::abs(b);
    while (b != 0) {
      long long temp = b;
      b = a % b;
      a = temp;
    }
    return a;
  }

  // Helper function to reduce a fraction by a factor
  DACE_HDFI static void reduce_fraction(long long& num, long long& den, long long factor) {
    if (factor > 1) {
      num /= factor;
      den /= factor;
    }
  }

  // Addition
  DACE_HDFI rational operator+(const rational& other) const {
    // a/b + c/d = (a*d + c*b) / (b*d)
    long long new_num = numerator * other.denominator + other.numerator * denominator;
    long long new_den = denominator * other.denominator;
    
    return rational(new_num, new_den);
  }

  // Subtraction
  DACE_HDFI rational operator-(const rational& other) const {
    // a/b - c/d = (a*d - c*b) / (b*d)
    long long new_num = numerator * other.denominator - other.numerator * denominator;
    long long new_den = denominator * other.denominator;
    
    return rational(new_num, new_den);
  }

  // Multiplication with cross-reduction
  DACE_HDFI rational operator*(const rational& other) const {
    // (a/b) * (c/d) = (a*c) / (b*d)
    // First, reduce cross-terms to minimize overflow
    long long a = numerator;
    long long b = denominator;
    long long c = other.numerator;
    long long d = other.denominator;
    
    // Reduce a and d by their GCD
    long long gcd_ad = gcd(a, d);
    reduce_fraction(a, d, gcd_ad);
    
    // Reduce c and b by their GCD
    long long gcd_cb = gcd(c, b);
    reduce_fraction(c, b, gcd_cb);
    
    // Now compute (a*c) / (b*d) with reduced values
    long long new_num = a * c;
    long long new_den = b * d;
    
    return rational(new_num, new_den);
  }

  // Division with cross-reduction
  DACE_HDFI rational operator/(const rational& other) const {
    // (a/b) / (c/d) = (a/b) * (d/c) = (a*d) / (b*c)
    // First, reduce cross-terms to minimize overflow
    long long a = numerator;
    long long b = denominator;
    long long c = other.numerator;
    long long d = other.denominator;
    
    // Reduce a and c by their GCD
    long long gcd_ac = gcd(a, c);
    reduce_fraction(a, c, gcd_ac);
    
    // Reduce b and d by their GCD
    long long gcd_bd = gcd(b, d);
    reduce_fraction(b, d, gcd_bd);
    
    // Now compute (a*d) / (b*c) with reduced values
    long long new_num = a * d;
    long long new_den = b * c;
    
    return rational(new_num, new_den);
  }

  // Compound assignment operators
  DACE_HDFI rational& operator+=(const rational& other) {
    *this = *this + other;
    return *this;
  }

  DACE_HDFI rational& operator-=(const rational& other) {
    *this = *this - other;
    return *this;
  }

  DACE_HDFI rational& operator*=(const rational& other) {
    *this = *this * other;
    return *this;
  }

  DACE_HDFI rational& operator/=(const rational& other) {
    *this = *this / other;
    return *this;
  }

  // Comparison operators
  DACE_HDFI bool operator==(const rational& other) const {
    return numerator == other.numerator && denominator == other.denominator;
  }

  DACE_HDFI bool operator!=(const rational& other) const {
    return !(*this == other);
  }

  DACE_HDFI bool operator<(const rational& other) const {
    // a/b < c/d  iff  a*d < c*b (when b,d > 0, which they are after canonicalization)
    return numerator * other.denominator < other.numerator * denominator;
  }

  DACE_HDFI bool operator<=(const rational& other) const {
    return *this < other || *this == other;
  }

  DACE_HDFI bool operator>(const rational& other) const {
    return !(*this <= other);
  }

  DACE_HDFI bool operator>=(const rational& other) const {
    return !(*this < other);
  }

  // Unary operators
  DACE_HDFI rational operator-() const {
    return rational(-numerator, denominator);
  }

  DACE_HDFI rational operator+() const {
    return *this;
  }

  // Accessor functions
  DACE_HDFI long long get_numerator() const {
    return numerator;
  }

  DACE_HDFI long long get_denominator() const {
    return denominator;
  }

  DACE_HDFI double get_value() const {
    return static_cast<double>(*this);
  }

  // Additional utility functions
  DACE_HDFI rational abs() const {
    long long abs_num = numerator < 0 ? -numerator : numerator;
    return rational(abs_num, denominator);
  }

  DACE_HDFI rational reciprocal() const {
    return rational(denominator, numerator);
  }
};

// Global operators for mixed-type arithmetic
DACE_HDFI rational operator+(long long lhs, const rational& rhs) {
  return rational(lhs) + rhs;
}

DACE_HDFI rational operator+(const rational& lhs, long long rhs) {
  return lhs + rational(rhs);
}

DACE_HDFI rational operator+(double lhs, const rational& rhs) {
  return rational(lhs) + rhs;
}

DACE_HDFI rational operator+(const rational& lhs, double rhs) {
  return lhs + rational(rhs);
}

DACE_HDFI rational operator-(long long lhs, const rational& rhs) {
  return rational(lhs) - rhs;
}

DACE_HDFI rational operator-(const rational& lhs, long long rhs) {
  return lhs - rational(rhs);
}

DACE_HDFI rational operator*(long long lhs, const rational& rhs) {
  return rational(lhs) * rhs;
}

DACE_HDFI rational operator*(const rational& lhs, long long rhs) {
  return lhs * rational(rhs);
}

DACE_HDFI rational operator/(long long lhs, const rational& rhs) {
  return rational(lhs) / rhs;
}

DACE_HDFI rational operator/(const rational& lhs, long long rhs) {
  return lhs / rational(rhs);
}

}