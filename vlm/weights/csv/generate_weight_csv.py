import pandas as pd
import os

def get_weight_input():
    """
    获取用户输入的权重组合并进行验证
    
    返回:
        list: 包含5个整数的权重列表 [waist, left_hand, right_hand, left_foot, right_foot]
    """
    while True:
        weight_input = input("\n请输入权重组合（5个数字，用逗号分隔，如：2,4,4,2,2）：").strip()
        
        # 验证输入格式
        try:
            # 分割输入并转换为整数
            weight_list = [int(num.strip()) for num in weight_input.split(',')]
            
            # 检查是否为5个数字
            if len(weight_list) != 5:
                print("❌ 错误：请输入恰好5个数字！")
                continue
            
            # 确认输入
            print(f"\n✅ 您输入的权重组合：{weight_list}")
            confirm = input("确认使用该组合？(y/n，默认y)：").strip().lower()
            if confirm in ['', 'y', 'yes']:
                return weight_list
            else:
                print("🔄 请重新输入...")
                
        except ValueError:
            print("❌ 错误：请输入有效的数字，并用逗号分隔！")
        except Exception as e:
            print(f"❌ 输入异常：{str(e)}")

def generate_weight_dataframe(weight_list):
    """
    根据权重列表生成包含权重数据的DataFrame
    
    参数:
        weight_list (list): 包含5个整数的权重列表
        
    返回:
        pandas.DataFrame: 包含权重数据的DataFrame
    """
    # 构建数据
    frame_numbers = list(range(181))  # frame列：0-180
    waist_vals = [weight_list[0]] * 181
    left_hand_vals = [weight_list[1]] * 181
    right_hand_vals = [weight_list[2]] * 181
    left_foot_vals = [weight_list[3]] * 181
    right_foot_vals = [weight_list[4]] * 181
    
    # 创建DataFrame
    df = pd.DataFrame({
        'frame': frame_numbers,
        'waist': waist_vals,
        'left_hand': left_hand_vals,
        'right_hand': right_hand_vals,
        'left_foot': left_foot_vals,
        'right_foot': right_foot_vals
    })
    
    return df

def save_weight_csv(df, weight_list):
    """
    保存权重数据到CSV文件
    
    参数:
        df (pandas.DataFrame): 要保存的DataFrame
        weight_list (list): 权重列表用于生成文件名
        
    返回:
        str: 保存的文件名
    """
    # 生成文件名（基于权重组合）
    weight_str = ''.join(map(str, weight_list))
    filename = f"weight_combat1_{weight_str}.csv"
    
    # 保存CSV
    df.to_csv(filename, index=False, encoding='utf-8')
    
    return filename

def show_result_info(filename, df):
    """
    显示生成结果信息和可选的数据预览
    
    参数:
        filename (str): 保存的文件名
        df (pandas.DataFrame): 生成的DataFrame数据
    """
    print("\n?? 生成完成！")
    print(f"📁 文件名：{filename}")
    print(f"📊 文件位置：{os.path.abspath(filename)}")
    print(f"📏 数据行数：{len(df)} 行")
    
    # 可选：显示前5行预览
    show_preview = input("\n是否查看前5行数据？(y/n，默认n)：").strip().lower()
    if show_preview in ['y', 'yes']:
        print("\n📋 数据预览（前5行）：")
        print(df.head())

def generate_weight_csv():
    """
    交互式生成权重组合CSV文件
    输入格式：5个数字的组合，用逗号分隔（如：2,4,4,2,2）
    对应列：waist, left_hand, right_hand, left_foot, right_foot
    """
    print("=" * 60)
    print("          权重组合CSV生成工具")
    print("=" * 60)
    
    # 1. 获取用户输入的权重组合
    weight_list = get_weight_input()
    
    # 2. 生成数据
    print("\n📝 正在生成CSV数据...")
    df = generate_weight_dataframe(weight_list)
    
    # 3. 保存文件
    filename = save_weight_csv(df, weight_list)
    
    # 4. 输出结果
    show_result_info(filename, df)

if __name__ == "__main__":
    try:
        generate_weight_csv()
    except KeyboardInterrupt:
        print("\n\n🔹 操作已取消")
    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}")
    finally:
        input("\n按回车键退出...")