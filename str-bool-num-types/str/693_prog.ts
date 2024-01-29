type Action =
  | { type: 'setTags', tags: string[] }
  | { type: 'closeTag', tag: string }
  | { type: 'showInput' }
  | { type: 'confirmInput', tag?: <FILL> }
  | { type: 'reset' }
  | { type: 'setIngredients', length: number }
  | { type: 'addIngredient' }
  | { type: 'removeIngredient', id: number }
  | { type: 'addDirection' }
  | { type: 'setDirections', length: number }
  | { type: 'removeDirection', id: number }

interface State {
  tags: string[]
  inputVisible: boolean
  ingredients: number[]
  directions: number[]
}


let ingredientId = 0
let directionId = 0
export const initialState = {
  tags: [],
  inputVisible: false,
  ingredients: [ingredientId],
  directions: [directionId],
}


const reducer = (state: State, action: Action) => {
  switch (action.type) {
    case 'setTags':
      return { ...state, tags: action.tags }
    case 'closeTag':
      return { ...state, tags: state.tags.filter(tag => tag !== action.tag) }

    case 'reset':
      return initialState

    case 'showInput':
      return { ...state, inputVisible: true }
    case 'confirmInput':
      let tag = action.tag
      if (!tag) return { ...state, inputVisible: false }
      tag = tag.trim()

      const { tags } = state
      let newTags = tags

      if (tag && tags.indexOf(tag) === -1) newTags = [...tags, tag]

      return { ...state, inputVisible: false, tags: newTags }

    case 'setIngredients':
      const ingredients = [] as number[]
      new Array(action.length).fill(null).forEach(() => {
        ingredientId += 1
        ingredients.push(ingredientId)
      })
      return { ...state, ingredients }
    case 'addIngredient':
      ingredientId += 1
      return {
        ...state,
        ingredients: [...state.ingredients, ingredientId],
      }
    case 'removeIngredient':
      return {
        ...state,
        ingredients: state.ingredients.filter(index => index !== action.id),
      }

    case 'setDirections':
      const directions = [] as number[]
      new Array(action.length).fill(null).forEach(() => {
        directionId += 1
        directions.push(ingredientId)
      })
      return { ...state, directions }
    case 'addDirection':
      directionId += 1
      return {
        ...state,
        directions: [...state.directions, directionId],
      }
    case 'removeDirection':
      return {
        ...state,
        directions: state.directions.filter(index => index !== action.id),
      }

    default:
      throw new Error('Unexpected action.')
  }
}


export default reducer
