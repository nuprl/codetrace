export const Kind = 'Core#ResponseAuthorization';

export interface IResponseAuthorization {
  kind: typeof Kind;
  /**
   * The requested by the authorization server authentication method
   */
  method: string;
  /**
   * The current state if the authorization process. This is used by NTLM authorization helper.
   */
  state?: number;
  /**
   * The headers association with the response.
   */
  headers?: string;
  /**
   * When returned by the server, the value of the challenge.
   */
  challengeHeader?: string;
}

export class ResponseAuthorization {
  kind = Kind;
  /**
   * The requested by the authorization server authentication method
   */
  method = 'unknown';
  /**
   * The current state if the authorization process. This is used by NTLM authorization helper.
   */
  state?: number;
  /**
   * The headers association with the response.
   */
  headers?: <FILL>;
  /**
   * When returned by the server, the value of the challenge.
   */
  challengeHeader?: string;

  /**
   * @param input The response authorization definition used to restore the state.
   */
  constructor(input?: string|IResponseAuthorization) {
    let init: IResponseAuthorization;
    if (typeof input === 'string') {
      init = JSON.parse(input);
    } else if (typeof input === 'object') {
      init = input;
    } else {
      init = {
        kind: Kind,
        method: 'unknown',
      };
    }
    this.new(init);
  }

  /**
   * Creates a new response authorization clearing anything that is so far defined.
   * 
   * Note, this throws an error when the object is not a response authorization.
   */
  new(init: IResponseAuthorization): void {
    if (!ResponseAuthorization.isResponseAuthorization(init)) {
      throw new Error(`Not a response authorization.`);
    }
    const { method, state, headers, challengeHeader } = init;
    this.kind = Kind;
    this.method = method;
    this.state = state;
    this.headers = headers;
    this.challengeHeader = challengeHeader;
  }

  /**
   * Checks whether the input is a definition of a response authorization.
   */
  static isResponseAuthorization(input: unknown): boolean {
    const typed = input as IResponseAuthorization;
    if (!input || !typed.method) {
      return false;
    }
    return true;
  }

  toJSON(): IResponseAuthorization {
    const result: IResponseAuthorization = {
      kind: Kind,
      method: this.method,
    };
    if (typeof this.state === 'number') {
      result.state = this.state;
    }
    if (this.headers) {
      result.headers = this.headers;
    }
    if (this.challengeHeader) {
      result.challengeHeader = this.challengeHeader;
    }
    return result;
  }
}
