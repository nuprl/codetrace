export const Kind = 'Core#Thing';

/**
 * An interface describing a base metadata of a thing.
 */
export declare interface IThing {
  /**
   * The data kind. The application ignores the input with an unknown `kind`, unless it can be determined from the context.
   */
  kind?: typeof Kind;
  /**
   * The name of the thing.
   */
  name?: string;
  /**
   * Optional value to overwrite the `name` in the UI.
   * The primary descriptive field is the `name`. The display name is only used in the presentation of the data.
   */
  displayName?: string;
  /**
   * The description of the thing.
   */
  description?: string;
  /**
   * The version number of the thing.
   */
  version?: string;
}

export class Thing {
  kind = Kind;
  /**
   * The name of the thing.
   */
  name?: string;
  /**
   * Optional value to overwrite the `name` in the UI.
   * The primary descriptive field is the `name`. The display name is only used in the presentation of the data.
   */
  displayName?: string;
  /**
   * The description of the thing.
   */
  description?: <FILL>;
  /**
   * The version number of the thing.
   */
  version?: string;

  /**
   * Returns one in this order:
   * - displayName
   * - name
   * - 'Unnamed object'
   */
  get renderLabel(): string {
    return this.displayName || this.name || 'Unnamed object';
  }

  /**
   * Creates a basic description from a name.
   */
  static fromName(name: string): Thing {
    const thing = new Thing({
      name,
      kind: Kind,
    });
    return thing;
  }

  /**
   * @param input The thing definition used to restore the state.
   */
  constructor(input?: string | IThing) {
    let init: IThing;
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
   * Creates a new thing clearing anything that is so far defined.
   * 
   * Note, this throws an error when the server is not a thing.
   */
  new(init: IThing): void {
    if (!Thing.isThing(init)) {
      throw new Error(`Not a thing.`);
    }
    const { description, name, version, displayName } = init;
    this.kind = Kind;
    this.name = name;
    this.displayName = displayName;
    this.description = description;
    this.version = version;
  }

  /**
   * Checks whether the input is a definition of a server.
   */
  static isThing(input: unknown): boolean {
    const typed = input as IThing;
    if (input && typed.kind === Kind) {
      return true;
    }
    return false;
  }

  toJSON(): IThing {
    const result: IThing = {
      kind: Kind,
    };
    if (typeof this.name === 'string') {
      result.name = this.name;
    }
    if (typeof this.displayName === 'string') {
      result.displayName = this.displayName;
    }
    if (this.description) {
      result.description = this.description;
    }
    if (this.version) {
      result.version = this.version;
    }
    return result;
  }
}
