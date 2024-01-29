export class Coin {
  id: number;
  name: string;
  symbol: string;
  slug: string;
  cmcRank: number;
  numMarketPairs: number;
  circulatingSupply: number;
  totalSupply: number;
  maxSupply: number;
  lastUpdated: Date;
  dateAdded: Date;
  tags: string[];
  platform: Platform;
  quote: Quotes;

  getColour(value: number): string {
    if (value > 0) {
      return 'green';
    }
    if (value < 0) {
      return 'red';
    }

    return '';
  }

  // we would like to map properties to javascript naming convention
  constructor(coin: ICoin) {
    // TODO: add defensive checks
    this.id = coin.id;
    this.name = coin.name;
    this.symbol = coin.symbol;
    this.slug = coin.slug;
    this.cmcRank = coin.cmc_rank;
    this.numMarketPairs = coin.num_market_pairs;
    this.circulatingSupply = coin.circulating_supply;
    this.totalSupply = coin.total_supply;
    this.maxSupply = coin.max_supply;
    this.lastUpdated = coin.last_updated;
    this.dateAdded = coin.date_added;
    this.tags = coin.tags;
    this.platform = coin.platform && new Platform(coin.platform);
    this.quote = coin.quote && new Quotes(coin.quote);
  }
}

export class Platform {
  id: number;
  name: string;
  symbol: <FILL>;
  slug: string;
  tokenAddress: string;

  constructor(platform: IPlatform) {
    this.id = platform.id;
    this.name = platform.name;
    this.symbol = platform.symbol;
    this.slug = platform.slug;
    this.tokenAddress = platform.token_address;
  }
}

export class Quotes {
  [key: string]: Quote;

  constructor(quotes: IQuotes) {
    Object.entries(quotes).forEach(([key, value]) => this[key] = new Quote(value));
  }
}

export class Quote {
  price: number;
  volume24h: number;
  volume24hReported: number;
  volume7d: number;
  volume7dReported: number;
  volume30d: number;
  volume30dReported: number;
  marketCap: number;
  change1h: number;
  change24h: number;
  change7d: number;
  lastUpdated: Date;

  constructor(quote: IQuote) {
    this.price = quote.price;
    this.volume24h = quote.volume_24h;
    this.volume24hReported = quote.volume_24h_reported;
    this.volume7d = quote.volume_7d;
    this.volume7dReported = quote.volume_7d_reported;
    this.volume30d = quote.volume_30d;
    this.volume30dReported = quote.volume_30d_reported;
    this.marketCap = quote.market_cap;
    this.change1h = quote.percent_change_1h / 100;
    this.change24h = quote.percent_change_24h / 100;
    this.change7d = quote.percent_change_7d / 100;
    this.lastUpdated = quote.last_updated;
  }
}

export interface IListingsResponse {
  data: ICoin[];
}

export interface IQuotesResponse {
  data: { [key: string]: ICoin; };
}

export interface ICoin {
  id: number;
  name: string;
  symbol: string;
  slug: string;
  cmc_rank: number;
  num_market_pairs: number;
  circulating_supply: number;
  total_supply: number;
  max_supply: number;
  last_updated: Date;
  date_added: Date;
  tags: string[];
  platform: IPlatform;
  quote: IQuotes;
}

export interface IPlatform {
  id: number;
  name: string;
  symbol: string;
  slug: string;
  token_address: string;
}

export interface IQuotes {
  [key: string]: IQuote;
}

export interface IQuote {
  price: number;
  volume_24h: number;
  volume_24h_reported: number;
  volume_7d: number;
  volume_7d_reported: number;
  volume_30d: number;
  volume_30d_reported: number;
  market_cap: number;
  percent_change_1h: number;
  percent_change_24h: number;
  percent_change_7d: number;
  last_updated: Date;
}
