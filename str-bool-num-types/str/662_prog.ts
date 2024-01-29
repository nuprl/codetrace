export const Kind = 'Core#License';

export interface ILicense {
  kind: typeof Kind;
  /**
   * The URL to the license text.
   * Only `url` or `content` can be used at a time.
   */
  url?: string;
  /**
   * The name of the license.
   */
  name?: string;
  /**
   * The content of the license.
   * Only `url` or `content` can be used at a time.
   */
  content?: <FILL>;
}

export class License {
  kind = Kind;
  /**
   * The URL to the license text.
   * Only `url` or `content` can be used at a time.
   */
  url?: string;
  /**
   * The name of the license.
   */
  name?: string;
  /**
   * The content of the license.
   * Only `url` or `content` can be used at a time.
   */
  content?: string;

  static fromUrl(url: string, name?: string): License {
    return new License({
      kind: Kind,
      url,
      name,
    });
  }

  static fromContent(content: string, name?: string): License {
    return new License({
      kind: Kind,
      content,
      name,
    });
  }

  /**
   * @param input The license definition used to restore the state.
   */
  constructor(input?: string|ILicense) {
    let init: ILicense;
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
   * Creates a new license clearing anything that is so far defined.
   * 
   * Note, this throws an error when the license is not a license object.
   */
  new(init: ILicense): void {
    if (!License.isLicense(init)) {
      throw new Error(`Not a license.`);
    }
    const { url, content, name } = init;
    this.kind = Kind;
    this.name = name;
    this.content = content;
    this.url = url;
  }

  /**
   * Checks whether the input is a definition of a license.
   */
  static isLicense(input: unknown): boolean {
    const typed = input as ILicense;
    if (typed && typed.kind === Kind) {
      return true;
    }
    return false;
  }

  toJSON(): ILicense {
    const result: ILicense = {
      kind: Kind,
    };
    if (this.name) {
      result.name = this.name;
    }
    if (this.url) {
      result.url = this.url;
    }
    if (this.content) {
      result.content = this.content;
    }
    return result;
  }
}
