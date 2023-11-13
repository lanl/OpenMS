#include "myclass.h"
#include <string>

MyClass::MyClass(const std::string &name, int value) : name(name), value(value) {}

void MyClass::setName(const std::string &name) {
    this->name = name;
}

std::string MyClass::getName() const {
    return name;
}

void MyClass::setValue(int value) {
    this->value = value;
}

int MyClass::getValue() const {
    return value;
}


/* //without using header
class MyClass {
public:
    MyClass(const std::string &name, int value) : name(name), value(value) {}

    void setName(const std::string &name) { this->name = name; }
    std::string getName() const { return name; }

    void setValue(int value) { this->value = value; }
    int getValue() const { return value; }

};
*/
