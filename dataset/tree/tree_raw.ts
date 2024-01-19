type Node_Type = Node_;
export class Node_{
	public data:any;
	public children: Node_Type[];
	public parent: Node_ | null;
	constructor(data: any){
		this.data = data;
		this.children = [];
		this.parent = null;
	}
}
export default class tree {
	public static head: Node_ = new Node_('head');
	public static path:string[][] | string[] = [];
	public static rootNode = '';
    
	static append(parent:any=undefined, data:any){
		const node = new Node_(data);
		if (!parent){
			this.head?.children.push(node);
			return;
		} else {
			const stack = [...this.head?.children] as Node_[];
			let prevIdx = null;
			while (stack?.length){
				const shiftItem = stack.shift();
				prevIdx = shiftItem;
				const children = shiftItem?.children;
				if (shiftItem?.data?.data === parent.data){
					node.parent = prevIdx as Node_;
					shiftItem?.children.push(node);
					return;
				} else {
					children?.forEach((node:Node_)=>{
						stack.push(node);
					});
				}
			}
		}
	}
	static getTree(){
		return this.head;
	}
	static async getPath(){
		// let rootNode = ;
		const rootNode = this.getRoot();
		const stack = [...this.head.children];
		let path_:string[] = [];
		while (stack.length){
			const shiftItem = stack.shift();
			path_.push(shiftItem?.data.data);
			const children = shiftItem?.children;
			if (shiftItem?.data.type === 'item'){
				(this.path as string[][]).push([...path_ as string[]]);
				if (shiftItem.parent){
					while (
						path_[path_.length - 1] !== rootNode?.data.data
					){
						path_.pop();
					}
				} else {
					path_ = [];
				}
			} else if (!shiftItem?.children.length){
				(this.path as string[][]).push([...path_ as string[]]);
				path_.pop();
				if (shiftItem?.parent){
					while (
						path_[path_.length - 1] !== rootNode?.data.data
					){
						path_.pop();
					}
				} else {
					path_ = [];
				}
			}
			for (let i=children?.length as number-1; i>=0; i--){
				stack.unshift(children?.[i] as Node_);
			}
		}
		return this.path;
	}
	private static getRoot(){
		const stack = [...tree.head.children];
		while (stack.length){
			const shiftItem = stack.shift();
			if (shiftItem?.data.type === 'parent'){
				return shiftItem;
			} else {
				const children = shiftItem?.children;
				for (let i=children?.length as number-1; i>=0; i--){
					stack.unshift(children?.[i] as Node_);
				}
			}
		}
		return undefined;
	}
	static clearTree(){
		this.head = new Node_('head');
		this.path = [];
		this.rootNode = '';
		return true;
	}
}

/**
 * a {
 *  b{
 *      d
 *   }
 *  c{
 *   }
 * }
 * 
 * 
 * 
 * 
 */
