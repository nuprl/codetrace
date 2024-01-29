interface IErrorOptions {
  cause?: Error;
}

export interface ISerializedError {
  message: string;
  stack?: string;
  code?: string | number;
  [key: PropertyKey]: any;
}

/**
 * An error that serializes!
 * It is a copy of the normal definition of an error but allows 
 * to serialize the value via the toJSON() function and restore previous values 
 * with the `new()` function.
 */
export class SerializableError {
  message: string;
  code?: number | string;
  private stackValue?: <FILL>;
  get stack(): string | undefined {
    return this.stackValue;
  }

  get name(): string {
    return 'SerializableError';
  }

  constructor();
  constructor(error: Error);
  constructor(message: string);
  constructor(message: string, options: IErrorOptions);
  constructor(message: string, code?: number | string);

  constructor(message?: string | Error, options: IErrorOptions | number | string = {}) {
    if (typeof message === 'string') {
      this.message = message;
    } else if (message) {
      this.message = message.message;
      this.stackValue = message.stack;
    } else {
      this.message = '';
    }
    if (typeof options === 'string' || typeof options === 'number') {
      this.code = options;
    } else if (options.cause && options.cause.stack) {
      this.stackValue = options.cause.stack;
    }
  }

  new(values: ISerializedError): void {
    if (values.message) {
      this.message = values.message;
    }
    if (values.stack) {
      this.stackValue = values.stack;
    }
    if (values.code || values.code === 0) {
      this.code = values.code;
    }
  }

  toJSON(): ISerializedError {
    const { message, stackValue: stack, code } = this;
    const result: ISerializedError = {
      message,
    };
    if (stack) {
      result.stack = stack;
    }
    if (code || code === 0) {
      result.code = code;
    }
    return result;
  }

  toString(): string {
    return this.message;
  }
}
