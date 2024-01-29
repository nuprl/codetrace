export const Kind = 'Core#HostRule';

/**
 * API Client host rule definition.
 */
export interface IHostRule {
  kind?: typeof Kind;
  /**
   * The from rule (may contain asterisks)
   */
  from: string;
  /**
   * replacement value
   */
  to: string;
  /**
   * if false the rule is ignored
   */
  enabled?: boolean;
  /**
   * optional rule description
   */
  comment?: string;
}

export class HostRule {
  kind = Kind;
  /**
   * The from rule (may contain asterisks)
   */
  from = '';
  /**
   * replacement value
   */
  to = '';
  /**
   * if false the rule is ignored
   */
  enabled?: boolean;
  /**
   * optional rule description
   */
  comment?: string;

  static fromValues(from: <FILL>, to: string ): HostRule {
    const result = new HostRule({
      kind: Kind,
      from,
      to,
      enabled: true,
    });
    return result;
  }

  /**
   * @param input The thing definition used to restore the state.
   */
  constructor(input?: string | IHostRule) {
    let init: IHostRule;
    if (typeof input === 'string') {
      init = JSON.parse(input);
    } else if (typeof input === 'object') {
      init = input;
    } else {
      init = {
        kind: Kind,
        from: '',
        to: '',
      };
    }
    this.new(init);
  }

  /**
   * Creates a new thing clearing anything that is so far defined.
   * 
   * Note, this throws an error when the server is not an API Client thing.
   */
  new(init: IHostRule): void {
    if (!HostRule.isHostRule(init)) {
      throw new Error(`Not a HostRule.`);
    }
    const { from='', to='', enabled, comment } = init;
    this.kind = Kind;
    this.from = from;
    this.to = to;
    this.enabled = enabled;
    this.comment = comment;
  }

  /**
   * Checks whether the input is a definition of a host rule.
   */
  static isHostRule(input: unknown): boolean {
    const typed = input as IHostRule;
    if (input && typed.kind && typed.kind === Kind) {
      return true;
    }
    return false;
  }

  toJSON(): IHostRule {
    const result: IHostRule = {
      kind: Kind,
      from : this.from,
      to : this.to,
    };
    if (typeof this.enabled === 'boolean') {
      result.enabled = this.enabled;
    }
    if (this.comment) {
      result.comment = this.comment;
    }
    return result;
  }

  /**
   * Applies `hosts` rules to the URL.
   *
   * @param value An URL to apply the rules to
   * @param rules List of host rules. It is a list of objects containing `from` and `to` properties.
   * @return Evaluated URL with hosts rules.
   */
  static applyHosts(value: string, rules: HostRule[]): string {
    if (!rules || !rules.length) {
      return value;
    }
    for (let i = 0; i < rules.length; i++) {
      const rule = rules[i];
      const result = HostRule.evaluateRule(value, rule);
      if (result) {
        value = result;
      }
    }
    return value;
  }

  /**
   * Evaluates hosts rule and applies it to the `url`.
   * @param {string} url The URL to evaluate
   * @param {HostRule} rule The host rule definition
   * @return {string} Processed url.
   */
  static evaluateRule(url: string, rule: HostRule): string | undefined {
    if (!rule || !rule.from || !rule.to) {
      return;
    }
    const re = HostRule.createRuleRe(rule.from);
    if (!re.test(url)) {
      return;
    }
    return url.replace(re, rule.to);
  }

  /**
   * @param input Rule body
   * @return Regular expression for the rule.
   */
  static createRuleRe(input: string): RegExp {
    input = input.replace(/\*/g, '(.*)');
    return new RegExp(input, 'gi');
  }
}
