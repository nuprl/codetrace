/* eslint-disable max-len */
// Send, Receive, Deposit, Withdrawal, Income
export interface Transaction {
    id: string;
    timestamp: Date;
    type: string;
    baseCurrency: string;
    baseQuantity: number;
    baseUsdPrice: number;
    feeCurrency: string;
    feeQuantity: number;
    feeUsdPrice: number;
    feeTotal: number;
    subTotal: number;
    total: number;
}

// Buy, Sell, Swap
export interface Order {
    id: string;
    timestamp: Date;
    type: string;
    baseCurrency: string;
    baseQuantity: number;
    baseUsdPrice: number;
    quoteCurrency: string;
    quoteQuantity: number;
    quotePrice: number;
    quoteUsdPrice: number;
    feeCurrency: string;
    feeQuantity: number;
    feeUsdPrice: number;
    feeTotal: number;
    subTotal: number;
    total: number;
}

/**
 * A base class to support general connection operations
 */
export default class BaseConnection {
    name: string;
    type: string;
    connection: any;
    credentials?: any;
    balances: any;
    symbols: Array<any>;
    initialized: boolean;
    requireSymbols: boolean;
    requireUsdValuation: boolean;
    stableCoins: Array<string> = ["USDC", "USDT"];
    fiatCurrencies: Array<string> = ["USD"];
    stableCurrencies: Array<string> = ["USD", "USDC", "USDT"];

    /**
     * Create BaseConnection instance
     */
    constructor(name: string, type: string, params: any = {}) {
        this.name = name;
        this.type = type;
        this.connection = null;
        this.initialized = false;
        this.symbols = [];
        this.requireSymbols = params.requireSymbols
            ? params.requireSymbols
            : false;
        this.requireUsdValuation = params.requireUsdValuation
            ? params.requireUsdValuation
            : true;
    }

    /**
     * JSON object representing connection
     */
    toJSON(): any {
        const baseJSON = {
            name: this.name,
            type: this.type,
            params: {},
        };
        if (this.credentials) {
            const updatedParams: any = baseJSON.params;
            updatedParams.credentials = this.credentials;
            baseJSON.params = updatedParams;
        }
        return baseJSON;
    }

    /**
     * HELPER: Convert buy/sell to swap if no fiat is involved in order
     */
    _attemptedSwapConversion(order: Order): Order {
        if (!this.fiatCurrencies.includes(order.quoteCurrency)) {
            if (order.type === "sell") {
                const tempBC = order.baseCurrency;
                const tempBQ = order.baseQuantity;
                const tempBP = order.baseUsdPrice;
                order.baseCurrency = order.quoteCurrency;
                order.baseQuantity = order.quoteQuantity;
                order.baseUsdPrice = order.quoteUsdPrice;
                order.quoteCurrency = tempBC;
                order.quoteQuantity = tempBQ;
                order.quoteUsdPrice = tempBP;
                order.quotePrice = order.baseUsdPrice / order.quoteUsdPrice;
            }
            order.type = "swap";
        }
        return order;
    }

    /**
     * Initialize exchange by fetching balances and loading markets
     */
    initialize(): <FILL> {
        throw Error(
            `NotImplementedError: ${this.name}.initialize() has not been implemented.`
        );
    }