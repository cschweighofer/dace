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
    lower = static_cast<float>(val - static_cast<double>(higher));
  }

 public:
  DACE_HDFI simulated_double() : higher(0.0f), lower(0.0f) {}

  DACE_HDFI simulated_double(double v) {
    decompose(v);
  }

  DACE_HDFI simulated_double& operator=(const simulated_double& other) {
    if (this != &other) {
      higher = other.higher;
      lower = other.lower;
    }
    return *this;
  }

  DACE_HDFI simulated_double(float v, float u) {
    this->higher = v;
    this->lower = u;
  }

  DACE_HDFI simulated_double(float v) : higher(v), lower(0.0f) {}

  DACE_HDFI simulated_double(int v) {
    decompose(static_cast<double>(v));
  }

  DACE_HDFI simulated_double(const simulated_double& other)
      : higher(other.higher), lower(other.lower) {}

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

  // Dekker's 1971 add2
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

    float z = r + s;
    float zz = r - z + s;

    return simulated_double(z, zz);
  }

  // Dekker's 1971 sub2
  DACE_HDFI simulated_double operator-(const simulated_double& other) const {
    float x = this->higher;
    float xx = this->lower;
    float y = other.higher;
    float yy = other.lower;

    float r, s;
    r = x - y;
    if (fabsf(x) > fabsf(y)) {
      s = x - r - y - yy + xx;
    } else {
      s = -y - r + x + xx - yy;
    }

    float z = r + s;
    float zz = r - z + s;

    return simulated_double(z, zz);
  }

  // Dekker's 1971 mul12 helper function
  DACE_HDFI simulated_double mul12(float x, float y) const {
    float hx, tx, hy, ty, p, q;
    float c = 23; //constant referring to mantissa bits of float
    p = x * c;
    hx = x - p + p; tx = x - hx;
    p = y * c;
    hy = y - p + p; ty = y - hy;
    p = hx * hy;
    q = hx * ty + tx * hy;

    float z = p + q;
    float zz = p - z + q + tx * ty;

    return simulated_double(z, zz);
  }

  //Dekkers 1971 mul2
  DACE_HDFI simulated_double operator*(const simulated_double& other) const {
    float x = this->higher;
    float xx = this->lower;
    float y = other.higher;
    float yy = other.lower;

    float c, cc;
    simulated_double product = mul12(x, y);
    c = product.lower;
    cc = product.higher;
    cc = x *yy + xx * y + cc;

    float z = c + cc;
    float zz = c - z + cc;

    return simulated_double(z, zz);
  }

  // Dekker's 1971 div2
  DACE_HDFI simulated_double operator/(const simulated_double& other) const {
    float x = this->higher;
    float xx = this->lower;
    float y = other.higher;
    float yy = other.lower;

    float c, cc, u, uu;
    c = x/y;
    simulated_double quotient = mul12(c, y);
    u = quotient.lower;
    uu = quotient.higher;
    cc = (x - u - uu + xx - c * yy) / y;

    float z = c + cc;
    float zz = c - z + cc;

    return simulated_double(z, zz);
  }

  // Dekker's 1971 sqrt for completeness
  DACE_HDFI simulated_double sqrt2(const simulated_double& other) const {
    float x = this->higher;
    float xx = this->lower;
    float y = other.higher;
    float yy = other.lower;

    float c, cc;
    simulated_double product = mul12(x, y);
    c = product.lower;
    cc = product.higher;
    cc = x * yy + xx * y + cc;

    float z = c + cc;
    float zz = c - z + cc;

    return simulated_double(z, zz);
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

DACE_HDFI simulated_double operator-(double lhs, const simulated_double& rhs) {
  return simulated_double(lhs) - rhs;
}

DACE_HDFI simulated_double operator-(const simulated_double& lhs, double rhs) {
  return lhs - simulated_double(rhs);
}

DACE_HDFI simulated_double operator*(double lhs, const simulated_double& rhs) {
  return simulated_double(lhs) * rhs;
}

DACE_HDFI simulated_double operator*(const simulated_double& lhs, double rhs) {
  return lhs * simulated_double(rhs);
}

DACE_HDFI simulated_double operator/(double lhs, const simulated_double& rhs) {
  return simulated_double(lhs) / rhs;
}

DACE_HDFI simulated_double operator/(const simulated_double& lhs, double rhs) {
  return lhs / simulated_double(rhs);
}

DACE_HDFI simulated_double operator+(float lhs, const simulated_double& rhs) {
  return simulated_double(lhs) + rhs;
}

DACE_HDFI simulated_double operator+(const simulated_double& lhs, float rhs) {
  return lhs + simulated_double(rhs);
}

DACE_HDFI simulated_double operator-(float lhs, const simulated_double& rhs) {
  return simulated_double(lhs) - rhs;
}

DACE_HDFI simulated_double operator-(const simulated_double& lhs, float rhs) {
  return lhs - simulated_double(rhs);
}

DACE_HDFI simulated_double operator*(float lhs, const simulated_double& rhs) {
  return simulated_double(lhs) * rhs;
}

DACE_HDFI simulated_double operator*(const simulated_double& lhs, float rhs) {
  return lhs * simulated_double(rhs);
}

DACE_HDFI simulated_double operator/(float lhs, const simulated_double& rhs) {
  return simulated_double(lhs) / rhs;
}

DACE_HDFI simulated_double operator/(const simulated_double& lhs, float rhs) {
  return lhs / simulated_double(rhs);
}

DACE_HDFI simulated_double operator+(int lhs, const simulated_double& rhs) {
  return simulated_double(lhs) + rhs;
}

DACE_HDFI simulated_double operator+(const simulated_double& lhs, int rhs) {
  return lhs + simulated_double(rhs);
}

DACE_HDFI simulated_double operator-(int lhs, const simulated_double& rhs) {
  return simulated_double(lhs) - rhs;
}

DACE_HDFI simulated_double operator-(const simulated_double& lhs, int rhs) {
  return lhs - simulated_double(rhs);
}

DACE_HDFI simulated_double operator*(int lhs, const simulated_double& rhs) {
  return simulated_double(lhs) * rhs;
}

DACE_HDFI simulated_double operator*(const simulated_double& lhs, int rhs) {
  return lhs * simulated_double(rhs);
}

DACE_HDFI simulated_double operator/(int lhs, const simulated_double& rhs) {
  return simulated_double(lhs) / rhs;
}

DACE_HDFI simulated_double operator/(const simulated_double& lhs, int rhs) {
  return lhs / simulated_double(rhs);
}

}