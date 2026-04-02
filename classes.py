class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

# Polymorphism in action:
# A single function that works for ANY animal
def make_it_talk(animal):
    print(animal.speak())

animals = [Dog(), Cat()]

for a in animals:
    make_it_talk(a) # Same function call, different results!