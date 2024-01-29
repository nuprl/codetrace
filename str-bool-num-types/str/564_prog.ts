/* eslint-disable @typescript-eslint/default-param-last */
export type Task = {
  id: string;
  title: string;
  description: string;
  date: Date;
  status: 'todo' | 'progress' | 'done';
  assignedTo: string;
};

interface TaskState {
  tasks: Task[];
  editableTask: Task;
}

const initialState: TaskState = {
  tasks: [],
  editableTask: {
    id: '',
    title: '',
    description: '',
    date: new Date(),
    status: 'todo',
    assignedTo: '',
  },
};

function taskReducer(
  state = initialState,
  action: { type: <FILL>; payload: any }
) {
  switch (action.type) {
    case 'ADD_TASK':
      return { ...state, tasks: [...state.tasks, action.payload] };
    case 'ADD_MANY_TASKS':
      return { ...state, tasks: action.payload };
    case 'PRE_EDIT':
      return {
        ...state,
        editableTask: { ...state.editableTask, ...action.payload },
      };
    case 'EDIT_TASK':
      return {
        ...state,
        tasks: state.tasks.map((task) => {
          if (task.id !== action.payload.id) return task;
          return { ...task, ...action.payload };
        }),
      };
    case 'MOVE_TASK':
      return {
        ...state,
        tasks: state.tasks.map((task) => {
          if (task.id !== action.payload.id) return task;
          return { ...task, status: action.payload.status };
        }),
      };
    case 'DELETE_TASK':
      return {
        ...state,
        tasks: state.tasks.filter((task) => {
          return task.id !== action.payload.id;
        }),
      };
    case 'DELETE_ALL_TASK':
      return { ...state, tasks: initialState.tasks };
    default:
      return state;
  }
}

export default taskReducer;
