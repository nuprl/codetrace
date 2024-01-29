// Product Tag
export type ProductTags = 'nike' | 'puma' | 'lifestyle' | 'caprese';

// Product Colors
export type ProductColor = 'white' | 'black' | 'red' | 'green' | 'purple' | 'yellow' | 'blue' | 'gray' | 'orange' | 'pink';


export class Product {
  id?: number;
  name?: string;
  price?: number;
  quantity?: number;
  type?: string;
  // tslint:disable-next-line:variable-name
  discount_price?: number;
  discount?: number;
  pictures?: string;
  // tslint:disable-next-line:variable-name
  image_url?: string;
  state?: string;
  // tslint:disable-next-line:variable-name
  is_used?: boolean;
  // tslint:disable-next-line:variable-name
  short_description?: string;
  description?: string;
  stock?: number;
  newPro?: boolean;
  brand?: string;
  sale?: boolean;
  category?: string;
  tags?: ProductTags[];
  colors?: ProductColor[];

  constructor(
    id?: number,
    name?: string,
    price?: number,
    quantity?: number,
    // tslint:disable-next-line:variable-name
    discount_price?: number,
    discount?: number,
    pictures?: string,
    type?: string,
    // tslint:disable-next-line:variable-name
    short_description?: string,
    description?: string,
    // tslint:disable-next-line:variable-name
    is_used?: <FILL>,
    stock?: number,
    state?: string,
    newPro?: boolean,
    brand?: string,
    sale?: boolean,
    category?: string,
    tags?: ProductTags[],
    colors?: ProductColor[]
  ) {
    this.id = id;
    this.name = name;
    this.price = price;
    this.quantity = quantity;
    this.type = type;
    this.is_used = is_used;
    this.discount_price = discount_price;
    this.discount = discount;
    this.pictures = pictures;
    this.short_description = short_description;
    this.description = description;
    this.stock = stock;
    this.newPro = newPro;
    this.brand = brand;
    this.sale = sale;
    this.category = category;
    this.tags = tags;
    this.colors = colors;
    this.state = state;
  }

}

// Color Filter
export interface ColorFilter {
  color?: ProductColor;
}
