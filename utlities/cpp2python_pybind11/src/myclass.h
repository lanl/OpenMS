#ifndef MYCLASS_H
#define MYCLASS_H

#include <string>

class MyClass {
public:
    MyClass(const std::string &name, int value);

    void setName(const std::string &name);
    std::string getName() const;

    void setValue(int value);
    int getValue() const;

private:
    std::string name;
    int value;
};

#endif // MYCLASS_H

