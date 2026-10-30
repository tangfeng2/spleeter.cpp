// Minimal JSON reader used to load Spleeter model configuration files.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace spleeter {

class JsonValue {
 public:
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Type type = Type::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string text;
  std::vector<JsonValue> items;
  std::map<std::string, JsonValue> members;

  bool IsNumber() const { return type == Type::kNumber; }
  bool IsString() const { return type == Type::kString; }
  bool IsArray() const { return type == Type::kArray; }
  bool IsObject() const { return type == Type::kObject; }

  // Returns nullptr when absent.
  const JsonValue* Find(const std::string& key) const;
  double GetNumber(double fallback) const;
  std::string GetString(const std::string& fallback) const;
  bool GetBool(bool fallback) const;
};

// Parses `text`. On failure returns false and fills `error`.
bool ParseJson(const std::string& text, JsonValue* out, std::string* error);

// Reads and parses a JSON file.
bool ReadJsonFile(const std::string& path, JsonValue* out, std::string* error);

}  // namespace spleeter
