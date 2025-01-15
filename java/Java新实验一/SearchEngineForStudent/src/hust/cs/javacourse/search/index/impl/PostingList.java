package hust.cs.javacourse.search.index.impl;

import hust.cs.javacourse.search.index.AbstractPosting;
import hust.cs.javacourse.search.index.AbstractPostingList;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PostingList extends AbstractPostingList {
    /**
     * 添加Posting,要求不能有内容重复的posting
     * @param posting：Posting对象
     */
    public void add(AbstractPosting posting){
         if(!this.list.contains(posting)){
             this.list.add(posting);
         }
    }

    /**
     * 获得PosingList的字符串表示
     * @return ： PosingList的字符串表示
     */
    @Override
    public String toString(){
        return this.list.toString();
    }

    /**
     * 添加Posting列表,,要求不能有内容重复的posting
     * @param postings：Posting列表
     */
    public void add(List<AbstractPosting> postings){
        for(AbstractPosting posting : postings){
            if(!this.list.contains(posting)){
                this.list.add(posting);
            }
        }
    }

    /**
     * 返回指定下标位置的Posting
     * @param index ：下标
     * @return： 指定下标位置的Posting
     */
    public AbstractPosting get(int index){
        return this.list.get(index);
    }

    /**
     * 返回指定Posting对象的下标
     * @param posting：指定的Posting对象
     * @return ：如果找到返回对应下标；否则返回-1
     */
    public int indexOf(AbstractPosting posting){
        return this.list.indexOf(posting);
    }

    /**
     * 返回指定文档id的Posting对象的下标
     * @param docId ：文档id
     * @return ：如果找到返回对应下标；否则返回-1
     */
    public int indexOf(int docId){
        for(int i=0;i<this.list.size();i++){
            if(this.list.get(i).getDocId() == docId){
                return i;
            }
        }
        return -1;
    }

    /**
     * 是否包含指定Posting对象
     * @param posting： 指定的Posting对象
     * @return : 如果包含返回true，否则返回false
     */
    public boolean contains(AbstractPosting posting){
        if(this.list.contains(posting)){
            return true;
        }
        return false;
    }

    /**
     * 删除指定下标的Posting对象
     * @param index：指定的下标
     */
    public void remove(int index){
        this.list.remove(index);
    }

    /**
     * 删除指定的Posting对象
     * @param posting ：定的Posting对象
     */
    public void remove(AbstractPosting posting){
        this.list.remove(posting);
    }

    /**
     * 返回PostingList的大小，即包含的Posting的个数
     * @return ：PostingList的大小
     */
    public int size(){
        return this.list.size();
    }

    /**
     * 清除PostingList
     */
    public void clear(){
        this.list.clear();
    }

    /**
     * PostingList是否为空
     * @return 为空返回true;否则返回false
     */
    public boolean isEmpty(){
        return this.list.isEmpty();
    }

    /**
     * 根据文档id的大小对PostingList进行从小到大的排序
     */
    public void sort(){
        Collections.sort(this.list);
    }

    @Override
    public void writeObject(ObjectOutputStream out) {
        try{
            out.writeObject(this.list);
        }catch(IOException e){
            e.printStackTrace();
        }
    }

    @Override
    public void readObject(ObjectInputStream in) {
        try{
            this.list=(ArrayList<AbstractPosting>)in.readObject();
            this.sort();
        }catch (IOException|ClassNotFoundException e){
            e.printStackTrace();
        }
    }
}
