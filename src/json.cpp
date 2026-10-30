#include "json.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace spleeter {

const JsonValue* JsonValue::Find(const std::string& key) const {
  if (type != Type::kObject) return nullptr;
  auto it = members.find(key);
  return it == members.end() ? nullptr : &it->second;
}

double JsonValue::GetNumber(double fallback) const {
  return type == Type::kNumber ? number : fallback;
}

std::string JsonValue::GetString(const std::string& fallback) const {
  return type == Type::kString ? text : fallback;
}

bool JsonValue::GetBool(bool fallback) const {
  return type == Type::kBool ? boolean : fallback;
}

namespace {

class Parser {
 public:
  Parser(const std::string& text) : text_(text) {}

  bool Parse(JsonValue* out, std::string* error) {
    SkipWhitespace();
    if (!ParseValue(out)) {
      *error = error_;
      return false;
    }
    SkipWhitespace();
    if (pos_ != text_.size()) {
      *error = "Trailing content at offset " + std::to_string(pos_);
      return false;
    }
    return true;
  }

 private:
  void SkipWhitespace() {
    while (pos_ < text_.size() &&
           (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' ||
            text_[pos_] == '\r')) {
      ++pos_;
    }
  }

  bool Fail(const std::string& message) {
    error_ = message + " at offset " + std::to_string(pos_);
    return false;
  }

  bool ParseValue(JsonValue* out) {
    if (pos_ >= text_.size()) return Fail("Unexpected end of input");
    const char c = text_[pos_];
    switch (c) {
      case '{':
        return ParseObject(out);
      case '[':
        return ParseArray(out);
      case '"': {
        out->type = JsonValue::Type::kString;
        return ParseString(&out->text);
      }
      case 't':
        if (text_.compare(pos_, 4, "true") == 0) {
          pos_ += 4;
          out->type = JsonValue::Type::kBool;
          out->boolean = true;
          return true;
        }
        return Fail("Invalid literal");
      case 'f':
        if (text_.compare(pos_, 5, "false") == 0) {
          pos_ += 5;
          out->type = JsonValue::Type::kBool;
          out->boolean = false;
          return true;
        }
        return Fail("Invalid literal");
      case 'n':
        if (text_.compare(pos_, 4, "null") == 0) {
          pos_ += 4;
          out->type = JsonValue::Type::kNull;
          return true;
        }
        return Fail("Invalid literal");
      default:
        return ParseNumber(out);
    }
  }

  bool ParseNumber(JsonValue* out) {
    const std::size_t start = pos_;
    if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) ++pos_;
    while (pos_ < text_.size() &&
           (std::isdigit(static_cast<unsigned char>(text_[pos_])) ||
            text_[pos_] == '.' || text_[pos_] == 'e' || text_[pos_] == 'E' ||
            text_[pos_] == '-' || text_[pos_] == '+')) {
      ++pos_;
    }
    if (start == pos_) return Fail("Invalid number");
    const std::string token = text_.substr(start, pos_ - start);
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str()) return Fail("Invalid number");
    out->type = JsonValue::Type::kNumber;
    out->number = value;
    return true;
  }

  bool ParseString(std::string* out) {
    if (pos_ >= text_.size() || text_[pos_] != '"') return Fail("Expected string");
    ++pos_;
    out->clear();
    while (pos_ < text_.size()) {
      const char c = text_[pos_++];
      if (c == '"') return true;
      if (c != '\\') {
        out->push_back(c);
        continue;
      }
      if (pos_ >= text_.size()) return Fail("Unterminated escape");
      const char esc = text_[pos_++];
      switch (esc) {
        case '"': out->push_back('"'); break;
        case '\\': out->push_back('\\'); break;
        case '/': out->push_back('/'); break;
        case 'b': out->push_back('\b'); break;
        case 'f': out->push_back('\f'); break;
        case 'n': out->push_back('\n'); break;
        case 'r': out->push_back('\r'); break;
        case 't': out->push_back('\t'); break;
        case 'u': {
          if (pos_ + 4 > text_.size()) return Fail("Bad \\u escape");
          const std::string hex = text_.substr(pos_, 4);
          pos_ += 4;
          const unsigned int code =
              static_cast<unsigned int>(std::strtoul(hex.c_str(), nullptr, 16));
          // Encode as UTF-8 (surrogate pairs are handled as-is).
          if (code < 0x80) {
            out->push_back(static_cast<char>(code));
          } else if (code < 0x800) {
            out->push_back(static_cast<char>(0xC0 | (code >> 6)));
            out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
          } else {
            out->push_back(static_cast<char>(0xE0 | (code >> 12)));
            out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
          }
          break;
        }
        default:
          return Fail("Unknown escape");
      }
    }
    return Fail("Unterminated string");
  }

  bool ParseArray(JsonValue* out) {
    out->type = JsonValue::Type::kArray;
    ++pos_;  // '['
    SkipWhitespace();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      return true;
    }
    while (true) {
      JsonValue item;
      SkipWhitespace();
      if (!ParseValue(&item)) return false;
      out->items.push_back(std::move(item));
      SkipWhitespace();
      if (pos_ >= text_.size()) return Fail("Unterminated array");
      if (text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (text_[pos_] == ']') {
        ++pos_;
        return true;
      }
      return Fail("Expected ',' or ']'");
    }
  }

  bool ParseObject(JsonValue* out) {
    out->type = JsonValue::Type::kObject;
    ++pos_;  // '{'
    SkipWhitespace();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return true;
    }
    while (true) {
      SkipWhitespace();
      std::string key;
      if (!ParseString(&key)) return false;
      SkipWhitespace();
      if (pos_ >= text_.size() || text_[pos_] != ':') return Fail("Expected ':'");
      ++pos_;
      SkipWhitespace();
      JsonValue value;
      if (!ParseValue(&value)) return false;
      out->members[key] = std::move(value);
      SkipWhitespace();
      if (pos_ >= text_.size()) return Fail("Unterminated object");
      if (text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (text_[pos_] == '}') {
        ++pos_;
        return true;
      }
      return Fail("Expected ',' or '}'");
    }
  }

  const std::string& text_;
  std::size_t pos_ = 0;
  std::string error_;
};

}  // namespace

bool ParseJson(const std::string& text, JsonValue* out, std::string* error) {
  Parser parser(text);
  return parser.Parse(out, error);
}

bool ReadJsonFile(const std::string& path, JsonValue* out, std::string* error) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    *error = "Cannot open JSON file: " + path;
    return false;
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return ParseJson(buffer.str(), out, error);
}

}  // namespace spleeter
