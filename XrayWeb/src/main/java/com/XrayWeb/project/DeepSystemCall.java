package com.XrayWeb.project;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class DeepSystemCall {
	public void linuxstart(String member_id, String file_name) throws IOException {
		System.out.println("System Call");
		System.out.println(member_id);
		System.out.println(file_name.substring(0,file_name.length()-4));
		String filter_filename = file_name.substring(0,file_name.length()-4);
		
		String cmd = "cd C:/Users/kenja/Desktop/XrayWeb/XrayWeb/src/main/webapp/resources/upload/file &&conda activate kdwtorch && python Xray_infer.py "+member_id+" "+filter_filename+" &"; 
	
		Process p = Runtime.getRuntime().exec("cmd /c " + cmd); 
		System.out.println(p);
		BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream())); 
		String l = null; 
		StringBuffer sb = new StringBuffer(); 
		sb.append(cmd); 
		while ((l = r.readLine()) != null) { 
			sb.append(l); sb.append("\n"); 
		}
		
	}
}