/**
 * An interface describing a provider of a thing.
 */
export declare interface IProvider {
  /**
   * The data kind. The application ignores the input with an unknown `kind`, unless it can be determined from the context.
   */
  kind: typeof Kind;
  /**
   * The URL to the provider
   */
  url?: string;
  /**
   * The name to the provider
   */
  name?: string;
  /**
   * The email to the provider
   */
  email?: <FILL>;
}

export const Kind = 'Core#Provider';

export class Provider {
  kind = Kind;
  /**
   * The URL to the provider
   */
  url?: string;
  /**
   * The name to the provider
   */
  name?: string;
  /**
   * The email to the provider
   */
  email?: string;
  /**
   * @param input The provider definition used to restore the state.
   */
  constructor(input?: string|IProvider) {
    let init: IProvider;
    if (typeof input === 'string') {
      init = JSON.parse(input);
    } else if (typeof input === 'object') {
      init = input;
    } else {
      init = {
        kind: Kind,
      };
    }
    this.new(init);
  }

  /**
   * Creates a new provider clearing anything that is so far defined.
   * 
   * Note, this throws an error when the provider is not a provider object.
   */
  new(init: IProvider): void {
    if (!Provider.isProvider(init)) {
      throw new Error(`Not a provider.`);
    }
    const { url, email, name } = init;
    this.kind = Kind;
    this.name = name;
    this.email = email;
    this.url = url;
  }

  /**
   * Checks whether the input is a definition of a provider.
   */
  static isProvider(input: unknown): boolean {
    const typed = input as IProvider;
    if (input && typed.kind === Kind) {
      return true;
    }
    return false;
  }

  toJSON(): IProvider {
    const result:IProvider = {
      kind: Kind,
    };
    if (this.url) {
      result.url = this.url;
    }
    if (this.email) {
      result.email = this.email;
    }
    if (this.name) {
      result.name = this.name;
    }
    return result;
  }
}
