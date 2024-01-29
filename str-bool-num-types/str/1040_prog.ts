export type chooseFoodCategory =
  | 'chicken'
  | 'pork'
  | 'beef'
  | 'minced'
  | 'salmon'
  | 'pike'
  | 'mackerel'
  | 'yellowtail'
  | 'miso'
  | 'ton'
  | 'vegetable'
  | 'vermicelli';

export type fetchData = {
  result: fetchFood[];
};

export type fetchFood = {
  foodImageUrl: string;
  mediumImageUrl: string;
  nickname: <FILL>;
  pickup: number;
  rank: string;
  recipeCost: string;
  recipeDescription: string;
  recipeId: number;
  recipeIndication: string;
  recipeMaterial: string[];
  recipePublishday: string;
  recipeTitle: string;
  recipeUrl: string;
  shop: number;
  smallImageUrl: string;
};

export type dataId =
  | '0,10-277'
  | '0,10-276'
  | '0,10-275'
  | '0,10-278'
  | '0,11-70'
  | '0,11-75'
  | '0,11-72'
  | '0,11-74'
  | '0,17-159'
  | '0,17-161'
  | '0,17-169'
  | '0,17-164-1369'
  | '1,10-277'
  | '1,10-276'
  | '1,10-275'
  | '1,10-278'
  | '1,11-70'
  | '1,11-75'
  | '1,11-72'
  | '1,11-74'
  | '1,17-159'
  | '1,17-161'
  | '1,17-169'
  | '1,17-164-1369'
  | '2,10-277'
  | '2,10-276'
  | '2,10-275'
  | '2,10-278'
  | '2,11-70'
  | '2,11-75'
  | '2,11-72'
  | '2,11-74'
  | '2,17-159'
  | '2,17-161'
  | '2,17-169'
  | '2,17-164-1369'
  | '3,10-277'
  | '3,10-276'
  | '3,10-275'
  | '3,10-278'
  | '3,11-70'
  | '3,11-75'
  | '3,11-72'
  | '3,11-74'
  | '3,17-159'
  | '3,17-161'
  | '3,17-169'
  | '3,17-164-1369';

export function throwIdFromCategory(category: chooseFoodCategory) {
  switch (category) {
    case 'chicken':
      return '10-277';
    case 'pork':
      return '10-276';
    case 'beef':
      return '10-275';
    case 'minced':
      return '10-278';
    case 'salmon':
      return '11-70';
    case 'pike':
      return '11-75';
    case 'mackerel':
      return '11-72';
    case 'yellowtail':
      return '11-74';
    case 'miso':
      return '17-159';
    case 'ton':
      return '17-161';
    case 'vegetable':
      return '17-169';
    case 'vermicelli':
      return '17-164-1369';
  }
}

export function throwNameFromCategory(category: string) {
  switch (category) {
    case 'meat':
      return '肉料理';
    case 'fish':
      return '魚料理';
    case 'soup':
      return 'スープ';
    case 'chicken':
      return '鶏肉料理';
    case 'pork':
      return '豚肉料理';
    case 'beef':
      return '牛肉料理';
    case 'minced':
      return '挽き肉料理';
    case 'salmon':
      return '鮭';
    case 'pike':
      return '秋刀魚';
    case 'mackerel':
      return '鯖';
    case 'yellowtail':
      return '鰤';
    case 'miso':
      return '味噌汁';
    case 'ton':
      return '豚汁';
    case 'vegetable':
      return '野菜スープ';
    case 'vermicelli':
      return '春雨スープ';
  }
}

export function validateString(str: string): boolean {
  const reg = new RegExp(/^[\u30a0-\u30ff\u3040-\u309f\u3005-\u3006\u30e0-\u9fcf]+$/);
  if (reg.test(str)) {
    return true;
  }
  return false;
}
