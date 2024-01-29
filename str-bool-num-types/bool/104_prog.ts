export type Constructor<T = any> = { new (...args: any[]): T };
export type ServiceRegistry = WeakMap<Constructor, any[]>;

export class Container {
  private registry: ServiceRegistry;
  private instances: WeakMap<Constructor, any>;

  constructor() {
    this.registry = new WeakMap();
    this.instances = new WeakMap();
  }

  /**
   * Adds a class and its constructor arguments to the container. Each time `get()` is called,
   * a new instance of the class will be returned.
   * @param newClass The class to add to the container.
   * @param args The arguments to pass to the class's constructor. Can be a primitive or
   * another class. If a class is used, the container will create an instance of it and
   * pass it to the constructor.
   * @returns The container object returns itself, allowing chaining multiple calls to `add()`.
   */
  add<T extends Constructor>(newClass: T, ...args: any[]): Container {
    this.registry.set(newClass, args);
    return this;
  }

  /**
   * Adds a static instance of a class to the container. Instead of creating a new instance
   * each time `get()` is called, that single instance will be returned instead.
   * @param className The class to add to the container.
   * @param args The arguments to pass to the class's constructor. Can be a primitive or
   * another class. If a class is used, the container will create an instance of it and
   * pass it to the constructor.
   * @returns The container object returns itself, allowing chaining multiple calls to `add()`.
   */
  addStatic<T extends Constructor>(className: T, ...args: any[]): Container {
    const instance = new className(...args);
    this.instances.set(className, instance);
    return this;
  }

  /**
   * Appends arguments to a class's existing arguments array.
   * @param className The class to modify.
   * @param newArgs The new arguments to append to the existing arguments array.
   */
  appendArgs(className: Constructor, ...newArgs: any[]): void {
    if (!this.has(className)) {
      throw new ContainerError(
        `${className.name} is not registered in container.`
      );
    }

    const args = this.registry.get(className) as any[]; // can never be undefined in `add()`
    args.push(...newArgs);
    this.registry.set(className, args);
  }

  /**
   * Reurns an instance of a class from the container. If the class was registered using
   * `addStatic()`, a singleton instance will be returned with each call to `get()`.
   *
   * If a class depends on another class added to the container, it will build an instance
   * of it using the stored arguments and inject it as a dependency. If a class was registered
   * using `addStatic()`, the container will inject a singleton instance of that class.
   * @param className The class to retrieve.
   * @returns An instance of the provided `className`.
   */
  get<T>(className: Constructor<T>): T {
    // Check for an existing static instance and return it.
    const existing = this.instances.get(className) as T;
    if (existing) {
      return existing;
    }

    // Otherwise, build a brand new instance.
    const args = this.registry.get(className);
    if (!args) {
      throw new ContainerError(
        `${className.name} is not registered in container.`
      );
    }

    const preparedArgs = args.map((arg) => {
      if (typeof arg != "function") {
        return arg;
      }

      // Try: is the function a constructor? If not, a TypeError will be thrown.
      try {
        new arg();
      } catch (error) {
        // In that case, return it as a callback argument.
        return arg;
      }

      // Otherwise, try to retrieve it from the container.
      const maybeConstructor = arg as Constructor;
      const registered = this.get(maybeConstructor);
      return registered;
    });

    return new className(...preparedArgs);
  }

  /**
   * Removes a class from the container.
   * @param className The class to remove.
   */
  delete(className: Constructor): void {
    if (!this.has(className)) {
      throw new ContainerError(
        `${className.name} is not registered in container.`
      );
    }
    this.registry.delete(className);
  }

  /**
   * Removes a static instance from the container.
   * @param className The class to remove.
   */
  deleteStatic(className: Constructor): void {
    if (!this.hasStatic(className)) {
      throw new ContainerError(
        `${className.name} is not registered in container.`
      );
    }
    this.instances.delete(className);
  }

  /**
   * Checks for the presence of a class in the registry.
   * @param className The class to look for.
   * @returns `true` if found, `false` if not.
   */
  has(className: Constructor): <FILL> {
    return this.registry.has(className);
  }

  /**
   * Checks for the presence of a static instance in the container.
   * @param className The class to look for.
   * @returns `true` if found, `false` if not.
   */
  hasStatic(className: Constructor): boolean {
    return this.instances.has(className);
  }
}

export class ContainerError extends Error {
  name = "ContainerError";

  constructor(public message: string) {
    super();
  }
}
