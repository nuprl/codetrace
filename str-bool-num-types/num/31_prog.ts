type Action =
  | { type: 'closeTag', tag: string }
  | { type: 'showInput' }
  | { type: 'confirmInput', tag?: string }
  | { type: 'reset' }
  | { type: 'addIngredient' }
  | { type: 'removeIngredient', id: number }
  | { type: 'addDirection' }
  | { type: 'removeDirection', id: <FILL> }

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
